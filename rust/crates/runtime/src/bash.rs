use std::env;
use std::io;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command as TokioCommand;
use tokio::runtime::Builder;
use tokio::time::timeout;

use crate::sandbox::{
    build_linux_sandbox_command, resolve_sandbox_status_for_request, FilesystemIsolationMode,
    SandboxConfig, SandboxStatus,
};
use crate::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BashCommandInput {
    pub command: String,
    pub timeout: Option<u64>,
    pub description: Option<String>,
    #[serde(rename = "run_in_background")]
    pub run_in_background: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "namespaceRestrictions")]
    pub namespace_restrictions: Option<bool>,
    #[serde(rename = "isolateNetwork")]
    pub isolate_network: Option<bool>,
    #[serde(rename = "filesystemMode")]
    pub filesystem_mode: Option<FilesystemIsolationMode>,
    #[serde(rename = "allowedMounts")]
    pub allowed_mounts: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BashCommandOutput {
    pub stdout: String,
    pub stderr: String,
    #[serde(rename = "rawOutputPath")]
    pub raw_output_path: Option<String>,
    pub interrupted: bool,
    #[serde(rename = "isImage")]
    pub is_image: Option<bool>,
    #[serde(rename = "backgroundTaskId")]
    pub background_task_id: Option<String>,
    #[serde(rename = "backgroundedByUser")]
    pub backgrounded_by_user: Option<bool>,
    #[serde(rename = "assistantAutoBackgrounded")]
    pub assistant_auto_backgrounded: Option<bool>,
    #[serde(rename = "dangerouslyDisableSandbox")]
    pub dangerously_disable_sandbox: Option<bool>,
    #[serde(rename = "returnCodeInterpretation")]
    pub return_code_interpretation: Option<String>,
    #[serde(rename = "noOutputExpected")]
    pub no_output_expected: Option<bool>,
    #[serde(rename = "structuredContent")]
    pub structured_content: Option<Vec<serde_json::Value>>,
    #[serde(rename = "persistedOutputPath")]
    pub persisted_output_path: Option<String>,
    #[serde(rename = "persistedOutputSize")]
    pub persisted_output_size: Option<u64>,
    #[serde(rename = "sandboxStatus")]
    pub sandbox_status: Option<SandboxStatus>,
}

pub fn execute_bash(input: BashCommandInput) -> io::Result<BashCommandOutput> {
    let cwd = env::current_dir()?;
    let sandbox_status = sandbox_status_for_input(&input, &cwd);

    if input.run_in_background.unwrap_or(false) {
        let mut child = prepare_command(&input.command, &cwd, &sandbox_status, false);
        let child = child
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        return Ok(BashCommandOutput {
            stdout: String::new(),
            stderr: String::new(),
            raw_output_path: None,
            interrupted: false,
            is_image: None,
            background_task_id: Some(child.id().to_string()),
            backgrounded_by_user: Some(false),
            assistant_auto_backgrounded: Some(false),
            dangerously_disable_sandbox: input.dangerously_disable_sandbox,
            return_code_interpretation: None,
            no_output_expected: Some(true),
            structured_content: None,
            persisted_output_path: None,
            persisted_output_size: None,
            sandbox_status: Some(sandbox_status),
        });
    }

    let runtime = Builder::new_current_thread().enable_all().build()?;
    runtime.block_on(execute_bash_async(input, sandbox_status, cwd))
}

async fn execute_bash_async(
    input: BashCommandInput,
    sandbox_status: SandboxStatus,
    cwd: std::path::PathBuf,
) -> io::Result<BashCommandOutput> {
    let timeout_ms = effective_timeout_ms(input.timeout);
    let mut command = prepare_tokio_command(&input.command, &cwd, &sandbox_status, true);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // Lead a fresh process group so a timeout can kill everything the command
    // started, including pipelines and `&` background jobs.
    #[cfg(unix)]
    command.process_group(0);

    let mut child = command.spawn()?;
    let process_group = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let completed = timeout(Duration::from_millis(timeout_ms), async {
        let (status, stdout, stderr) =
            tokio::join!(child.wait(), read_capped(stdout), read_capped(stderr));
        Ok::<_, io::Error>((status?, stdout?, stderr?))
    })
    .await;

    let (status, stdout, stderr) = match completed {
        Ok(Ok(collected)) => collected,
        Ok(Err(error)) => {
            kill_process_group(&mut child, process_group).await;
            return Err(error);
        }
        Err(_) => {
            kill_process_group(&mut child, process_group).await;
            return Ok(BashCommandOutput {
                stdout: String::new(),
                stderr: format!("Command exceeded timeout of {timeout_ms} ms"),
                raw_output_path: None,
                interrupted: true,
                is_image: None,
                background_task_id: None,
                backgrounded_by_user: None,
                assistant_auto_backgrounded: None,
                dangerously_disable_sandbox: input.dangerously_disable_sandbox,
                return_code_interpretation: Some(String::from("timeout")),
                no_output_expected: Some(true),
                structured_content: None,
                persisted_output_path: None,
                persisted_output_size: None,
                sandbox_status: Some(sandbox_status),
            });
        }
    };

    let interrupted = false;
    let stdout = truncate_output(&String::from_utf8_lossy(&stdout));
    let stderr = truncate_output(&String::from_utf8_lossy(&stderr));
    let no_output_expected = Some(stdout.trim().is_empty() && stderr.trim().is_empty());
    let return_code_interpretation = status.code().and_then(|code| {
        if code == 0 {
            None
        } else {
            Some(format!("exit_code:{code}"))
        }
    });

    Ok(BashCommandOutput {
        stdout,
        stderr,
        raw_output_path: None,
        interrupted,
        is_image: None,
        background_task_id: None,
        backgrounded_by_user: None,
        assistant_auto_backgrounded: None,
        dangerously_disable_sandbox: input.dangerously_disable_sandbox,
        return_code_interpretation,
        no_output_expected,
        structured_content: None,
        persisted_output_path: None,
        persisted_output_size: None,
        sandbox_status: Some(sandbox_status),
    })
}

/// Timeout applied when the caller does not pass one (matches upstream).
const DEFAULT_TIMEOUT_MS: u64 = 120_000;
/// Longest timeout a caller may request (matches upstream).
const MAX_TIMEOUT_MS: u64 = 600_000;
/// Bytes kept per output stream. The rest is drained and dropped so a chatty
/// command (`yes`) cannot grow memory without bound before its timeout.
const MAX_CAPTURE_BYTES: usize = 4 * MAX_OUTPUT_BYTES;

fn effective_timeout_ms(requested: Option<u64>) -> u64 {
    requested.unwrap_or(DEFAULT_TIMEOUT_MS).min(MAX_TIMEOUT_MS)
}

async fn read_capped<R>(reader: Option<R>) -> io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut captured = Vec::new();
    let Some(mut reader) = reader else {
        return Ok(captured);
    };
    let mut buffer = vec![0_u8; 8192];
    loop {
        let read = reader.read(&mut buffer).await?;
        if read == 0 {
            return Ok(captured);
        }
        let room = MAX_CAPTURE_BYTES.saturating_sub(captured.len());
        captured.extend_from_slice(&buffer[..read.min(room)]);
    }
}

/// Kill the command's whole process group, then the direct child, and reap it.
/// `process_group` is the child's pid captured at spawn time, because tokio
/// forgets the pid once the direct child has been reaped while background
/// jobs in its group are still holding the output pipes open.
async fn kill_process_group(child: &mut tokio::process::Child, process_group: Option<u32>) {
    #[cfg(unix)]
    if let Some(pgid) = process_group.and_then(|pid| i32::try_from(pid).ok()) {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(pgid),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    #[cfg(not(unix))]
    let _ = process_group;
    let _ = child.start_kill();
    let _ = child.wait().await;
}

fn sandbox_status_for_input(input: &BashCommandInput, cwd: &std::path::Path) -> SandboxStatus {
    let config = ConfigLoader::default_for(cwd).load().map_or_else(
        |_| SandboxConfig::default(),
        |runtime_config| runtime_config.sandbox().clone(),
    );
    let request = config.resolve_request(
        input.dangerously_disable_sandbox.map(|disabled| !disabled),
        input.namespace_restrictions,
        input.isolate_network,
        input.filesystem_mode,
        input.allowed_mounts.clone(),
    );
    resolve_sandbox_status_for_request(&request, cwd)
}

fn prepare_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> Command {
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    if let Some(launcher) = build_linux_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = Command::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.envs(launcher.env);
        return prepared;
    }

    let mut prepared = Command::new("sh");
    prepared.arg("-c").arg(command).current_dir(cwd);
    if sandbox_status.filesystem_active {
        prepared.env("HOME", cwd.join(".sandbox-home"));
        prepared.env("TMPDIR", cwd.join(".sandbox-tmp"));
    }
    prepared
}

fn prepare_tokio_command(
    command: &str,
    cwd: &std::path::Path,
    sandbox_status: &SandboxStatus,
    create_dirs: bool,
) -> TokioCommand {
    if create_dirs {
        prepare_sandbox_dirs(cwd);
    }

    if let Some(launcher) = build_linux_sandbox_command(command, cwd, sandbox_status) {
        let mut prepared = TokioCommand::new(launcher.program);
        prepared.args(launcher.args);
        prepared.current_dir(cwd);
        prepared.envs(launcher.env);
        return prepared;
    }

    let mut prepared = TokioCommand::new("sh");
    prepared.arg("-c").arg(command).current_dir(cwd);
    if sandbox_status.filesystem_active {
        prepared.env("HOME", cwd.join(".sandbox-home"));
        prepared.env("TMPDIR", cwd.join(".sandbox-tmp"));
    }
    prepared
}

fn prepare_sandbox_dirs(cwd: &std::path::Path) {
    let _ = std::fs::create_dir_all(cwd.join(".sandbox-home"));
    let _ = std::fs::create_dir_all(cwd.join(".sandbox-tmp"));
}

#[cfg(test)]
mod tests {
    use super::{execute_bash, BashCommandInput};
    use crate::sandbox::FilesystemIsolationMode;

    #[test]
    fn executes_simple_command() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(false),
            namespace_restrictions: Some(false),
            isolate_network: Some(false),
            filesystem_mode: Some(FilesystemIsolationMode::WorkspaceOnly),
            allowed_mounts: None,
        })
        .expect("bash command should execute");

        assert_eq!(output.stdout, "hello");
        assert!(!output.interrupted);
        assert!(output.sandbox_status.is_some());
    }

    #[test]
    fn disables_sandbox_when_requested() {
        let output = execute_bash(BashCommandInput {
            command: String::from("printf 'hello'"),
            timeout: Some(1_000),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: Some(true),
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
        })
        .expect("bash command should execute");

        assert!(!output.sandbox_status.expect("sandbox status").enabled);
    }

    #[cfg(unix)]
    #[test]
    fn timeout_kills_the_command_and_its_background_children() {
        let _guard = crate::test_env_lock();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time should move forward")
            .as_nanos();
        let marker = std::env::temp_dir().join(format!("bash-timeout-marker-{unique}"));
        let started = std::time::Instant::now();

        let output = execute_bash(BashCommandInput {
            command: format!("(sleep 1; touch '{}') & sleep 5", marker.display()),
            timeout: Some(200),
            description: None,
            run_in_background: Some(false),
            dangerously_disable_sandbox: None,
            namespace_restrictions: None,
            isolate_network: None,
            filesystem_mode: None,
            allowed_mounts: None,
        })
        .expect("bash command should execute");

        assert!(output.interrupted);
        assert!(started.elapsed() < std::time::Duration::from_secs(4));
        std::thread::sleep(std::time::Duration::from_millis(1_500));
        assert!(
            !marker.exists(),
            "a process started by the timed-out command kept running"
        );
    }

    #[test]
    fn applies_default_and_maximum_timeouts() {
        assert_eq!(super::effective_timeout_ms(None), 120_000);
        assert_eq!(super::effective_timeout_ms(Some(250)), 250);
        assert_eq!(super::effective_timeout_ms(Some(u64::MAX)), 600_000);
    }

    #[test]
    fn caps_captured_output_while_draining_the_stream() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("runtime should build");
        let source = vec![b'x'; 1_000_000];

        let captured = runtime
            .block_on(super::read_capped(Some(source.as_slice())))
            .expect("read should succeed");

        assert_eq!(captured.len(), super::MAX_CAPTURE_BYTES);
    }
}

/// Maximum output bytes before truncation (16 KiB, matching upstream).
const MAX_OUTPUT_BYTES: usize = 16_384;

/// Truncate output to `MAX_OUTPUT_BYTES`, appending a marker when trimmed.
fn truncate_output(s: &str) -> String {
    if s.len() <= MAX_OUTPUT_BYTES {
        return s.to_string();
    }
    // Find the last valid UTF-8 boundary at or before MAX_OUTPUT_BYTES
    let mut end = MAX_OUTPUT_BYTES;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut truncated = s[..end].to_string();
    truncated.push_str("\n\n[output truncated — exceeded 16384 bytes]");
    truncated
}

#[cfg(test)]
mod truncation_tests {
    use super::*;

    #[test]
    fn short_output_unchanged() {
        let s = "hello world";
        assert_eq!(truncate_output(s), s);
    }

    #[test]
    fn long_output_truncated() {
        let s = "x".repeat(20_000);
        let result = truncate_output(&s);
        assert!(result.len() < 20_000);
        assert!(result.ends_with("[output truncated — exceeded 16384 bytes]"));
    }

    #[test]
    fn exact_boundary_unchanged() {
        let s = "a".repeat(MAX_OUTPUT_BYTES);
        assert_eq!(truncate_output(&s), s);
    }

    #[test]
    fn one_over_boundary_truncated() {
        let s = "a".repeat(MAX_OUTPUT_BYTES + 1);
        let result = truncate_output(&s);
        assert!(result.contains("[output truncated"));
    }
}
