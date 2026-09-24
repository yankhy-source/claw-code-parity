//! Minimal shell command-line splitting for permission decisions.
//!
//! Two deliberately different views of a command line:
//!
//! * [`split_simple_commands`] is strict and quote-aware. It is meant for
//!   checks that *grant* something (allow rules, read-only classification)
//!   and returns `None` for any construct whose effect cannot be read off the
//!   text, so callers fail closed.
//! * [`command_segments`] is aggressive and ignores quoting. It is meant for
//!   checks that *restrict* something (deny and ask rules) and surfaces
//!   commands hidden in chains, substitutions, subshells or quoted `sh -c`
//!   payloads. It is best effort: no string matcher can see through
//!   `$(printf rm)`, aliases, functions or scripts.

/// Leading words that run the command after them. A restrictive segment is
/// also checked with these stripped, so `sudo rm -rf ~` is seen as `rm -rf ~`.
const COMMAND_PREFIX_WORDS: &[&str] = &[
    "!", "builtin", "command", "do", "doas", "elif", "else", "env", "exec", "if", "nice", "nohup",
    "sudo", "then", "time", "until", "while", "xargs",
];

/// Split `command` into simple commands (argv word lists, quotes removed)
/// separated by `;`, `&`, `&&`, `|`, `||` or newlines.
///
/// Returns `None` when the line uses anything whose effect cannot be
/// determined from the text alone: parameter or command expansion (`$`,
/// backticks), redirections (`<`, `>`), subshells and groups (`(`, `)`, `{`,
/// `}`), backslash escapes, unterminated quotes or control characters.
pub(crate) fn split_simple_commands(command: &str) -> Option<Vec<Vec<String>>> {
    let mut commands = Vec::new();
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut chars = command.chars();

    while let Some(ch) = chars.next() {
        match ch {
            '\'' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '\'' => break,
                        other => word.push(other),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '"' => break,
                        '$' | '`' | '\\' => return None,
                        other => word.push(other),
                    }
                }
            }
            ' ' | '\t' => finish_word(&mut words, &mut word, &mut in_word),
            '\n' | ';' | '&' | '|' => {
                finish_word(&mut words, &mut word, &mut in_word);
                if !words.is_empty() {
                    commands.push(std::mem::take(&mut words));
                }
            }
            '$' | '`' | '\\' | '<' | '>' | '(' | ')' | '{' | '}' => return None,
            other if other.is_control() => return None,
            other => {
                in_word = true;
                word.push(other);
            }
        }
    }

    finish_word(&mut words, &mut word, &mut in_word);
    if !words.is_empty() {
        commands.push(words);
    }
    Some(commands)
}

fn finish_word(words: &mut Vec<String>, word: &mut String, in_word: &mut bool) {
    if *in_word {
        words.push(std::mem::take(word));
        *in_word = false;
    }
}

/// Every plausible command start in `command`, normalized with
/// [`normalize_command_text`]. The line is cut at every shell control,
/// grouping, quoting, redirection and expansion character; each piece is
/// returned as is and again after stripping each leading `NAME=value`
/// assignment or wrapper word such as `sudo`, `env` or `then`.
pub(crate) fn command_segments(command: &str) -> Vec<String> {
    let cleaned = command.replace('\\', "");
    let mut segments = Vec::new();
    for piece in cleaned.split(is_segment_boundary) {
        let mut words = piece.split_whitespace().collect::<Vec<_>>();
        while let Some(first) = words.first().copied() {
            segments.push(normalize_words(&words));
            if is_assignment(first) || COMMAND_PREFIX_WORDS.contains(&basename(first)) {
                words.remove(0);
            } else {
                break;
            }
        }
    }
    segments
}

/// Collapse whitespace and reduce the first word to its basename, so
/// `/bin/rm  -rf ~` and `rm -rf ~` compare equal.
pub(crate) fn normalize_command_text(text: &str) -> String {
    normalize_words(&text.split_whitespace().collect::<Vec<_>>())
}

fn normalize_words(words: &[&str]) -> String {
    let mut normalized = String::new();
    for (index, word) in words.iter().enumerate() {
        if index == 0 {
            normalized.push_str(basename(word));
        } else {
            normalized.push(' ');
            normalized.push_str(word);
        }
    }
    normalized
}

pub(crate) fn basename(word: &str) -> &str {
    word.rsplit('/').next().unwrap_or(word)
}

fn is_segment_boundary(ch: char) -> bool {
    matches!(
        ch,
        ';' | '&' | '|' | '\n' | '\r' | '(' | ')' | '`' | '{' | '}' | '\'' | '"' | '<' | '>' | '$'
    )
}

fn is_assignment(word: &str) -> bool {
    word.split_once('=').is_some_and(|(name, _)| {
        name.chars()
            .next()
            .is_some_and(|first| !first.is_ascii_digit())
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    })
}

#[cfg(test)]
mod tests {
    use super::{command_segments, split_simple_commands};

    #[test]
    fn strict_split_handles_separators_and_quotes() {
        assert_eq!(
            split_simple_commands("git status && git commit -m 'a; b' | cat"),
            Some(vec![
                vec!["git".to_string(), "status".to_string()],
                vec![
                    "git".to_string(),
                    "commit".to_string(),
                    "-m".to_string(),
                    "a; b".to_string()
                ],
                vec!["cat".to_string()],
            ])
        );
        assert_eq!(
            split_simple_commands("echo ''"),
            Some(vec![vec!["echo".to_string(), String::new()]])
        );
    }

    #[test]
    fn strict_split_rejects_unresolvable_constructs() {
        for command in [
            "echo $HOME",
            "echo `id`",
            "echo \"$(id)\"",
            "cat < in",
            "echo x > out",
            "(ls)",
            "{ ls; }",
            "echo a\\;b",
            "echo 'unterminated",
            "ls\rrm",
        ] {
            assert_eq!(split_simple_commands(command), None, "{command:?}");
        }
    }

    #[test]
    fn restrictive_segments_surface_hidden_commands() {
        let segments = command_segments("FOO=1 sudo /bin/rm -rf ~; echo \"$(curl x)\"");
        for expected in ["rm -rf ~", "sudo /bin/rm -rf ~", "curl x"] {
            assert!(
                segments.iter().any(|segment| segment == expected),
                "{expected:?} missing from {segments:?}"
            );
        }
    }
}
