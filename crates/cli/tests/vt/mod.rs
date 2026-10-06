//! A tiny terminal emulator for tests: it replays the bytes a program wrote to a pseudo-terminal and says what
//! the screen would show, including autowrap at the last column, so layout bugs show up as wrong rows.

use unicode_width::UnicodeWidthChar;

pub struct Screen {
    cols: usize,
    rows: Vec<Vec<char>>,
    /// Stands in for a row that does not exist, so writes past the end are dropped instead of failing.
    spare: Vec<char>,
    row: usize,
    col: usize,
}

impl Screen {
    pub fn new(cols: usize) -> Self {
        Self {
            cols,
            rows: vec![Vec::new()],
            spare: Vec::new(),
            row: 0,
            col: 0,
        }
    }

    fn line(&mut self, row: usize) -> &mut Vec<char> {
        while self.rows.len() <= row {
            self.rows.push(Vec::new());
        }
        self.rows.get_mut(row).unwrap_or(&mut self.spare)
    }

    fn put(&mut self, c: char) {
        let width = UnicodeWidthChar::width(c).unwrap_or(0);
        if width == 0 {
            return;
        }
        if self.col >= self.cols {
            self.row = self.row.saturating_add(1);
            self.col = 0;
        }
        let (row, col) = (self.row, self.col);
        let line = self.line(row);
        while line.len() < col {
            line.push(' ');
        }
        line.truncate(col);
        line.push(c);
        if width == 2 {
            line.push('\0');
        }
        self.col = self.col.saturating_add(width);
    }

    fn erase_line(&mut self, mode: usize) {
        let (row, col) = (self.row, self.col);
        let line = self.line(row);
        if mode == 0 {
            line.truncate(col);
        } else {
            line.clear();
        }
    }

    fn csi(&mut self, params: &str, last: char) {
        if params.starts_with('?') {
            return;
        }
        let numbers: Vec<usize> = params.split(';').map(|p| p.parse().unwrap_or(0)).collect();
        let first = numbers.first().copied().unwrap_or(0);
        let count = first.max(1);
        match last {
            'A' => self.row = self.row.saturating_sub(count),
            'B' => self.row = self.row.saturating_add(count),
            'C' => self.col = self.col.saturating_add(count),
            'D' => self.col = self.col.saturating_sub(count),
            'G' => self.col = first.max(1).saturating_sub(1),
            'H' | 'f' => {
                self.row = first.max(1).saturating_sub(1);
                self.col = numbers
                    .get(1)
                    .copied()
                    .unwrap_or(1)
                    .max(1)
                    .saturating_sub(1);
            }
            'K' => self.erase_line(first),
            'J' if first == 2 || first == 3 => {
                self.rows = vec![Vec::new()];
            }
            'J' => {
                self.erase_line(0);
                self.rows.truncate(self.row.saturating_add(1));
            }
            _ => {}
        }
    }

    pub fn feed(&mut self, bytes: &[u8]) {
        let text = String::from_utf8_lossy(bytes);
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '\x1b' => match chars.peek() {
                    Some('[') => {
                        chars.next();
                        let mut params = String::new();
                        for next in chars.by_ref() {
                            if next.is_ascii_alphabetic() || next == '~' {
                                self.csi(&params, next);
                                break;
                            }
                            params.push(next);
                        }
                    }
                    Some(_) => {
                        chars.next();
                    }
                    None => {}
                },
                '\r' => self.col = 0,
                '\n' => {
                    self.row = self.row.saturating_add(1);
                    let row = self.row;
                    self.line(row);
                }
                '\x08' => self.col = self.col.saturating_sub(1),
                c if c.is_control() => {}
                c => self.put(c),
            }
        }
    }

    /// The rows as text, without trailing spaces and without trailing empty rows.
    pub fn lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .rows
            .iter()
            .map(|r| {
                r.iter()
                    .filter(|c| **c != '\0')
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect();
        while lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        lines
    }

    /// Row and column of the cursor.
    pub fn cursor(&self) -> (usize, usize) {
        (self.row, self.col)
    }
}
