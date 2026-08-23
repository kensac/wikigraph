//! Streaming lexer for MySQL dump `INSERT INTO ... VALUES (...),(...);` lines.
//!
//! mysqldump escapes newlines inside strings, so a physical line never splits
//! a row; we read line-by-line and lex tuples out of INSERT statements.

use std::borrow::Cow;
use std::io::BufRead;

use anyhow::{bail, Context, Result};

#[derive(Debug, PartialEq)]
pub enum Value<'a> {
    Int(i64),
    Float(f64),
    Str(Cow<'a, [u8]>),
    Null,
}

impl<'a> Value<'a> {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Str(v) => Some(v),
            _ => None,
        }
    }
}

/// Streams every row of every INSERT statement to `f`. Returns row count.
/// `f` may return `false` to stop early.
pub fn for_each_row<R, F>(mut reader: R, mut f: F) -> Result<u64>
where
    R: BufRead,
    F: FnMut(&[Value]) -> bool,
{
    let mut line: Vec<u8> = Vec::with_capacity(1 << 20);
    let mut rows = 0u64;

    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(rows);
        }
        if !line.starts_with(b"INSERT INTO") {
            continue;
        }
        let mut pos = match memchr::memchr(b'(', &line) {
            Some(p) => p,
            None => continue,
        };
        let mut vals: Vec<Value> = Vec::with_capacity(16);
        loop {
            debug_assert_eq!(line[pos], b'(');
            pos += 1;
            vals.clear();
            loop {
                let (v, next) = lex_value(&line, pos)?;
                vals.push(v);
                pos = next;
                match line.get(pos) {
                    Some(b',') => pos += 1,
                    Some(b')') => {
                        pos += 1;
                        break;
                    }
                    other => bail!("unexpected byte after value: {other:?}"),
                }
            }
            rows += 1;
            if !f(&vals) {
                return Ok(rows);
            }
            // after tuple: `,(` continues the statement, anything else ends it
            if line.get(pos) == Some(&b',') && line.get(pos + 1) == Some(&b'(') {
                pos += 1;
            } else {
                break;
            }
        }
    }
}

fn lex_value(line: &[u8], pos: usize) -> Result<(Value<'_>, usize)> {
    match *line.get(pos).context("truncated tuple")? {
        b'\'' => lex_string(line, pos + 1),
        b'N' => {
            if line[pos..].starts_with(b"NULL") {
                Ok((Value::Null, pos + 4))
            } else {
                bail!("bad literal at {pos}")
            }
        }
        _ => lex_number(line, pos),
    }
}

fn lex_string(line: &[u8], start: usize) -> Result<(Value<'_>, usize)> {
    match memchr::memchr2(b'\'', b'\\', &line[start..]) {
        // fast path: closes with no escapes, borrow directly
        Some(off) if line[start + off] == b'\'' => {
            let end = start + off;
            Ok((Value::Str(Cow::Borrowed(&line[start..end])), end + 1))
        }
        Some(_) => lex_string_owned(line, start),
        None => bail!("unterminated string at {start}"),
    }
}

fn lex_string_owned(line: &[u8], start: usize) -> Result<(Value<'_>, usize)> {
    let mut out = Vec::with_capacity(32);
    let mut pos = start;
    loop {
        match memchr::memchr2(b'\'', b'\\', &line[pos..]) {
            Some(off) => {
                out.extend_from_slice(&line[pos..pos + off]);
                pos += off;
                if line[pos] == b'\'' {
                    return Ok((Value::Str(Cow::Owned(out)), pos + 1));
                }
                let esc = *line.get(pos + 1).context("truncated escape")?;
                out.push(match esc {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'0' => 0,
                    b'Z' => 0x1a,
                    other => other,
                });
                pos += 2;
            }
            None => bail!("unterminated string at {start}"),
        }
    }
}

fn lex_number(line: &[u8], start: usize) -> Result<(Value<'_>, usize)> {
    let mut pos = start;
    while pos < line.len() && line[pos] != b',' && line[pos] != b')' {
        pos += 1;
    }
    let text = std::str::from_utf8(&line[start..pos]).context("non-utf8 number")?;
    if let Ok(v) = text.parse::<i64>() {
        return Ok((Value::Int(v), pos));
    }
    let v: f64 = text
        .parse()
        .with_context(|| format!("bad number {text:?}"))?;
    Ok((Value::Float(v), pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    enum OwnedValue {
        Int(i64),
        Float(f64),
        Str(Vec<u8>),
        Null,
    }

    impl From<&Value<'_>> for OwnedValue {
        fn from(v: &Value) -> Self {
            match v {
                Value::Int(i) => OwnedValue::Int(*i),
                Value::Float(f) => OwnedValue::Float(*f),
                Value::Str(s) => OwnedValue::Str(s.to_vec()),
                Value::Null => OwnedValue::Null,
            }
        }
    }

    fn collect(input: &str) -> Vec<Vec<OwnedValue>> {
        let mut rows = Vec::new();
        for_each_row(input.as_bytes(), |vals| {
            rows.push(vals.iter().map(OwnedValue::from).collect());
            true
        })
        .unwrap();
        rows
    }

    #[test]
    fn basic_rows() {
        let rows = collect("INSERT INTO `page` VALUES (1,0,'Foo',0.5,NULL),(2,-3,'B',1,7);\n");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], OwnedValue::Int(1));
        assert_eq!(rows[0][2], OwnedValue::Str(b"Foo".to_vec()));
        assert_eq!(rows[0][3], OwnedValue::Float(0.5));
        assert_eq!(rows[0][4], OwnedValue::Null);
        assert_eq!(rows[1][1], OwnedValue::Int(-3));
    }

    #[test]
    fn escapes() {
        let rows = collect(r"INSERT INTO `t` VALUES ('It\'s','a\\b','x\ny','q(z)');");
        assert_eq!(rows[0][0], OwnedValue::Str(b"It's".to_vec()));
        assert_eq!(rows[0][1], OwnedValue::Str(b"a\\b".to_vec()));
        assert_eq!(rows[0][2], OwnedValue::Str(b"x\ny".to_vec()));
        assert_eq!(rows[0][3], OwnedValue::Str(b"q(z)".to_vec()));
    }

    #[test]
    fn skips_non_insert_lines() {
        let input = "-- comment\nCREATE TABLE `x` (id int);\nINSERT INTO `x` VALUES (42);\n";
        let rows = collect(input);
        assert_eq!(rows, vec![vec![OwnedValue::Int(42)]]);
    }

    #[test]
    fn early_stop() {
        let mut seen = 0;
        let n = for_each_row("INSERT INTO `t` VALUES (1),(2),(3);".as_bytes(), |_| {
            seen += 1;
            seen < 2
        })
        .unwrap();
        assert_eq!(n, 2);
    }
}
