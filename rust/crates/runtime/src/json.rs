use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// Deeper input is rejected instead of recursing until the stack overflows.
const MAX_NESTING_DEPTH: usize = 128;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(i64),
    /// A number with a fraction or exponent, or an integer outside the `i64` range.
    Float(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

// The parser only produces finite floats, for which `==` is an equivalence relation.
impl Eq for JsonValue {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError {
    message: String,
}

impl JsonError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for JsonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}

impl JsonValue {
    #[must_use]
    pub fn render(&self) -> String {
        match self {
            Self::Null => "null".to_string(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => value.to_string(),
            Self::Float(value) => render_float(*value),
            Self::String(value) => render_string(value),
            Self::Array(values) => {
                let rendered = values
                    .iter()
                    .map(Self::render)
                    .collect::<Vec<_>>()
                    .join(",");
                format!("[{rendered}]")
            }
            Self::Object(entries) => {
                let rendered = entries
                    .iter()
                    .map(|(key, value)| format!("{}:{}", render_string(key), value.render()))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{{rendered}}}")
            }
        }
    }

    pub fn parse(source: &str) -> Result<Self, JsonError> {
        let mut parser = Parser::new(source);
        let value = parser.parse_value()?;
        parser.skip_whitespace();
        if parser.is_eof() {
            Ok(value)
        } else {
            Err(JsonError::new("unexpected trailing content"))
        }
    }

    #[must_use]
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }
}

fn render_float(value: f64) -> String {
    if value.is_finite() {
        // Debug keeps a fraction or exponent, so the value re-parses as a float.
        format!("{value:?}")
    } else {
        // JSON has no NaN or infinity; the parser never produces them.
        "null".to_string()
    }
}

fn render_string(value: &str) -> String {
    let mut rendered = String::with_capacity(value.len() + 2);
    rendered.push('"');
    for ch in value.chars() {
        match ch {
            '"' => rendered.push_str("\\\""),
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            '\u{08}' => rendered.push_str("\\b"),
            '\u{0C}' => rendered.push_str("\\f"),
            control if control.is_control() => push_unicode_escape(&mut rendered, control),
            plain => rendered.push(plain),
        }
    }
    rendered.push('"');
    rendered
}

fn push_unicode_escape(rendered: &mut String, control: char) {
    const HEX: &[u8; 16] = b"0123456789abcdef";

    rendered.push_str("\\u");
    let value = u32::from(control);
    for shift in [12_u32, 8, 4, 0] {
        let nibble = ((value >> shift) & 0xF) as usize;
        rendered.push(char::from(HEX[nibble]));
    }
}

struct Parser<'a> {
    chars: Vec<char>,
    index: usize,
    depth: usize,
    _source: &'a str,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().collect(),
            index: 0,
            depth: 0,
            _source: source,
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonError> {
        self.skip_whitespace();
        match self.peek() {
            Some('n') => self.parse_literal("null", JsonValue::Null),
            Some('t') => self.parse_literal("true", JsonValue::Bool(true)),
            Some('f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some('"') => self.parse_string().map(JsonValue::String),
            Some('[') => self.parse_array(),
            Some('{') => self.parse_object(),
            Some('-' | '0'..='9') => self.parse_number(),
            Some(other) => Err(JsonError::new(format!("unexpected character: {other}"))),
            None => Err(JsonError::new("unexpected end of input")),
        }
    }

    fn parse_literal(&mut self, expected: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        for expected_char in expected.chars() {
            if self.next() != Some(expected_char) {
                return Err(JsonError::new(format!(
                    "invalid literal: expected {expected}"
                )));
            }
        }
        Ok(value)
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect('"')?;
        let mut value = String::new();
        while let Some(ch) = self.next() {
            match ch {
                '"' => return Ok(value),
                '\\' => value.push(self.parse_escape()?),
                plain => value.push(plain),
            }
        }
        Err(JsonError::new("unterminated string"))
    }

    fn parse_escape(&mut self) -> Result<char, JsonError> {
        match self.next() {
            Some('"') => Ok('"'),
            Some('\\') => Ok('\\'),
            Some('/') => Ok('/'),
            Some('b') => Ok('\u{08}'),
            Some('f') => Ok('\u{0C}'),
            Some('n') => Ok('\n'),
            Some('r') => Ok('\r'),
            Some('t') => Ok('\t'),
            Some('u') => self.parse_unicode_escape(),
            Some(other) => Err(JsonError::new(format!("invalid escape sequence: {other}"))),
            None => Err(JsonError::new("unexpected end of input in escape sequence")),
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char, JsonError> {
        let mut value = self.parse_hex_quad()?;
        // Characters outside the BMP are escaped as a UTF-16 surrogate pair.
        if (0xD800..=0xDBFF).contains(&value) {
            if !(self.try_consume('\\') && self.try_consume('u')) {
                return Err(JsonError::new("unpaired surrogate in unicode escape"));
            }
            let low = self.parse_hex_quad()?;
            if !(0xDC00..=0xDFFF).contains(&low) {
                return Err(JsonError::new("unpaired surrogate in unicode escape"));
            }
            value = 0x10000 + ((value - 0xD800) << 10) + (low - 0xDC00);
        }
        char::from_u32(value).ok_or_else(|| JsonError::new("invalid unicode scalar value"))
    }

    fn parse_hex_quad(&mut self) -> Result<u32, JsonError> {
        let mut value = 0_u32;
        for _ in 0..4 {
            let Some(ch) = self.next() else {
                return Err(JsonError::new("unexpected end of input in unicode escape"));
            };
            value = (value << 4)
                | ch.to_digit(16)
                    .ok_or_else(|| JsonError::new("invalid unicode escape"))?;
        }
        Ok(value)
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonError> {
        self.expect('[')?;
        self.enter_nested()?;
        let mut values = Vec::new();
        loop {
            self.skip_whitespace();
            if self.try_consume(']') {
                break;
            }
            values.push(self.parse_value()?);
            self.skip_whitespace();
            if self.try_consume(']') {
                break;
            }
            self.expect(',')?;
        }
        self.depth -= 1;
        Ok(JsonValue::Array(values))
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonError> {
        self.expect('{')?;
        self.enter_nested()?;
        let mut entries = BTreeMap::new();
        loop {
            self.skip_whitespace();
            if self.try_consume('}') {
                break;
            }
            let key = self.parse_string()?;
            self.skip_whitespace();
            self.expect(':')?;
            let value = self.parse_value()?;
            entries.insert(key, value);
            self.skip_whitespace();
            if self.try_consume('}') {
                break;
            }
            self.expect(',')?;
        }
        self.depth -= 1;
        Ok(JsonValue::Object(entries))
    }

    fn enter_nested(&mut self) -> Result<(), JsonError> {
        self.depth += 1;
        if self.depth > MAX_NESTING_DEPTH {
            return Err(JsonError::new(format!(
                "maximum nesting depth of {MAX_NESTING_DEPTH} exceeded"
            )));
        }
        Ok(())
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let mut value = String::new();
        if self.try_consume('-') {
            value.push('-');
        }
        if self.consume_digits(&mut value) == 0 {
            return Err(JsonError::new("invalid number"));
        }

        let mut is_float = false;
        if self.try_consume('.') {
            value.push('.');
            if self.consume_digits(&mut value) == 0 {
                return Err(JsonError::new("invalid number: expected digit after '.'"));
            }
            is_float = true;
        }
        if let Some(exponent @ ('e' | 'E')) = self.peek() {
            self.index += 1;
            value.push(exponent);
            if let Some(sign @ ('+' | '-')) = self.peek() {
                self.index += 1;
                value.push(sign);
            }
            if self.consume_digits(&mut value) == 0 {
                return Err(JsonError::new("invalid number: expected digit in exponent"));
            }
            is_float = true;
        }

        if !is_float {
            if let Ok(integer) = value.parse::<i64>() {
                return Ok(JsonValue::Number(integer));
            }
        }
        value
            .parse::<f64>()
            .ok()
            .filter(|number| number.is_finite())
            .map(JsonValue::Float)
            .ok_or_else(|| JsonError::new("number out of range"))
    }

    fn consume_digits(&mut self, value: &mut String) -> usize {
        let start = self.index;
        while let Some(ch @ '0'..='9') = self.peek() {
            value.push(ch);
            self.index += 1;
        }
        self.index - start
    }

    fn expect(&mut self, expected: char) -> Result<(), JsonError> {
        match self.next() {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(JsonError::new(format!(
                "expected '{expected}', found '{actual}'"
            ))),
            None => Err(JsonError::new(format!(
                "expected '{expected}', found end of input"
            ))),
        }
    }

    fn try_consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(' ' | '\n' | '\r' | '\t')) {
            self.index += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.index).copied()
    }

    fn next(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.index += 1;
        Some(ch)
    }

    fn is_eof(&self) -> bool {
        self.index >= self.chars.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{render_string, JsonValue};
    use std::collections::BTreeMap;

    #[test]
    fn renders_and_parses_json_values() {
        let mut object = BTreeMap::new();
        object.insert("flag".to_string(), JsonValue::Bool(true));
        object.insert(
            "items".to_string(),
            JsonValue::Array(vec![
                JsonValue::Number(4),
                JsonValue::String("ok".to_string()),
            ]),
        );

        let rendered = JsonValue::Object(object).render();
        let parsed = JsonValue::parse(&rendered).expect("json should parse");

        assert_eq!(parsed.as_object().expect("object").len(), 2);
    }

    #[test]
    fn escapes_control_characters() {
        assert_eq!(render_string("a\n\t\"b"), "\"a\\n\\t\\\"b\"");
    }

    #[test]
    fn parses_decimal_and_exponent_numbers() {
        let parsed = JsonValue::parse(
            r#"{"timeout": 1.5, "scale": 1e3, "ratio": -2.5E-2, "big": 18446744073709551616, "retries": 3}"#,
        )
        .expect("valid JSON numbers should parse");
        let object = parsed.as_object().expect("object");

        assert_eq!(object["retries"].as_i64(), Some(3));
        assert_eq!(object["timeout"].render(), "1.5");
        assert_eq!(object["scale"].render(), "1000.0");
        assert_eq!(object["ratio"].render(), "-0.025");
        assert_eq!(object["big"].render(), "1.8446744073709552e19");
        assert_eq!(object["timeout"].as_i64(), None);
        assert_eq!(
            JsonValue::parse(&parsed.render()).expect("rendered floats should re-parse"),
            parsed
        );
    }

    #[test]
    fn rejects_malformed_and_non_finite_numbers() {
        for source in ["1.", "-", "1e", "1e+", ".5", "1e400", "-1.5e999"] {
            assert!(
                JsonValue::parse(source).is_err(),
                "{source} should be rejected"
            );
        }
    }

    #[test]
    fn decodes_surrogate_pair_escapes() {
        let parsed = JsonValue::parse(r#""smile \ud83d\ude00 \u00e4""#).expect("should parse");
        assert_eq!(parsed.as_str(), Some("smile \u{1F600} \u{e4}"));

        for lone in [
            r#""\ud83d""#,
            r#""\ud83d x""#,
            r#""\ude00""#,
            r#""\ud83d\u0041""#,
        ] {
            assert!(JsonValue::parse(lone).is_err(), "{lone} should be rejected");
        }
    }

    #[test]
    fn limits_nesting_depth_instead_of_overflowing_the_stack() {
        let shallow = format!("{}{}", "[".repeat(100), "]".repeat(100));
        assert!(JsonValue::parse(&shallow).is_ok());

        let deep_arrays = "[".repeat(100_000);
        let deep_objects = r#"{"a":"#.repeat(100_000);
        for deep in [deep_arrays, deep_objects] {
            let error = JsonValue::parse(&deep).expect_err("deep nesting should be rejected");
            assert!(error.to_string().contains("nesting"), "{error}");
        }
    }
}
