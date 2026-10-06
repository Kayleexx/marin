use unicode_width::UnicodeWidthStr;

use super::theme::{Theme, looks};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

/// A table collected row by row; it is laid out once, when it ends.
#[derive(Debug, Default)]
pub struct Table {
    pub header: Vec<String>,
    pub aligns: Vec<Align>,
    pub rows: Vec<Vec<String>>,
}

/// Cells of a `| a | b |` line; escaped bars stay inside their cell.
pub fn cells(line: &str) -> Vec<String> {
    let mut cells = vec![String::new()];
    let mut escaped = false;
    for c in line.trim().chars() {
        match (c, escaped) {
            ('|', false) => cells.push(String::new()),
            ('\\', false) => escaped = true,
            _ => {
                escaped = false;
                if let Some(cell) = cells.last_mut() {
                    cell.push(c);
                }
            }
        }
    }
    if cells.first().is_some_and(|c| c.trim().is_empty()) {
        cells.remove(0);
    }
    if cells.last().is_some_and(|c| c.trim().is_empty()) {
        cells.pop();
    }
    cells.into_iter().map(|c| c.trim().to_string()).collect()
}

/// The alignments of a `|---|:--:|` delimiter row, or `None` if the line is not one.
pub fn delimiter(line: &str) -> Option<Vec<Align>> {
    let parts = cells(line);
    if parts.is_empty() {
        return None;
    }
    parts
        .iter()
        .map(|cell| {
            let body = cell.trim_matches(':');
            let dashes = !body.is_empty() && body.chars().all(|c| c == '-');
            let (left, right) = (cell.starts_with(':'), cell.ends_with(':'));
            dashes.then_some(match (left, right) {
                (true, true) => Align::Center,
                (false, true) => Align::Right,
                _ => Align::Left,
            })
        })
        .collect()
}

/// Cell text without inline markers, which cannot be styled once the column widths are fixed.
fn plain(cell: &str) -> String {
    let mut text = cell.replace("**", "").replace("__", "").replace('`', "");
    while let (Some(open), Some(close)) = (text.find('['), text.find("](")) {
        let Some(end) = text.get(close..).and_then(|t| t.find(')')) else {
            break;
        };
        let label = text
            .get(open.saturating_add(1)..close)
            .unwrap_or("")
            .to_string();
        let tail = text
            .get(close.saturating_add(end).saturating_add(1)..)
            .unwrap_or("")
            .to_string();
        text = format!("{}{label}{tail}", text.get(..open).unwrap_or(""));
    }
    text
}

/// Splits `text` into lines of at most `width` columns, breaking at spaces and inside overlong words.
pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        let mut word = word.to_string();
        let sep = usize::from(!line.is_empty());
        if line
            .width()
            .saturating_add(sep)
            .saturating_add(word.width())
            <= width
        {
            if sep == 1 {
                line.push(' ');
            }
            line.push_str(&word);
            continue;
        }
        if !line.is_empty() {
            lines.push(std::mem::take(&mut line));
        }
        while word.width() > width {
            let mut head = String::new();
            let mut rest = String::new();
            for c in word.chars() {
                if rest.is_empty()
                    && head
                        .width()
                        .saturating_add(UnicodeWidthStr::width(c.to_string().as_str()))
                        <= width
                {
                    head.push(c);
                } else {
                    rest.push(c);
                }
            }
            lines.push(head);
            word = rest;
        }
        line = word;
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

/// Narrowest column a table may be squeezed to.
const MIN_COLUMN: usize = 4;

/// Column widths that fit `room`, shrinking the widest columns first; `None` if even narrow columns do not fit.
fn fit_widths(natural: &[usize], room: usize) -> Option<Vec<usize>> {
    let overhead = natural.len().saturating_mul(3).saturating_add(1);
    let budget = room.saturating_sub(overhead);
    let mut widths = natural.to_vec();
    while widths.iter().sum::<usize>() > budget {
        let (index, widest) = widths
            .iter()
            .enumerate()
            .max_by_key(|(_, w)| **w)
            .map(|(i, w)| (i, *w))?;
        if widest <= MIN_COLUMN {
            return None;
        }
        *widths.get_mut(index)? = widest.saturating_sub(1);
    }
    Some(widths)
}

fn pad(text: &str, width: usize, align: Align) -> String {
    let gap = width.saturating_sub(text.width());
    match align {
        Align::Left => format!("{text}{}", " ".repeat(gap)),
        Align::Right => format!("{}{text}", " ".repeat(gap)),
        Align::Center => {
            let left = gap / 2;
            format!(
                "{}{text}{}",
                " ".repeat(left),
                " ".repeat(gap.saturating_sub(left))
            )
        }
    }
}

/// Lines (styled, with their display widths) for the table, or `None` when it cannot fit `room`
/// and should be shown another way.
pub fn layout(table: &Table, room: usize, theme: &Theme) -> Option<Vec<(String, usize)>> {
    let count = table.header.len();
    let mut rows: Vec<Vec<String>> = vec![table.header.iter().map(|c| plain(c)).collect()];
    for row in &table.rows {
        rows.push(
            (0..count)
                .map(|i| plain(row.get(i).map_or("", String::as_str)))
                .collect(),
        );
    }
    let natural: Vec<usize> = (0..count)
        .map(|i| {
            rows.iter()
                .map(|r| r.get(i).map_or(0, |c| c.width()))
                .max()
                .unwrap_or(0)
                .max(1)
        })
        .collect();
    let widths = fit_widths(&natural, room)?;
    let frame = theme.frame();
    let rule = |parts: [&str; 3]| {
        let inner: Vec<String> = widths
            .iter()
            .map(|w| frame.horizontal.repeat(w.saturating_add(2)))
            .collect();
        format!("{}{}{}", parts[0], inner.join(parts[1]), parts[2])
    };
    let bar = theme.paint(&looks::DIM, frame.vertical);
    let mut out: Vec<(String, usize)> = Vec::new();
    let total = widths
        .iter()
        .sum::<usize>()
        .saturating_add(count.saturating_mul(3))
        .saturating_add(1);
    out.push((theme.paint(&looks::DIM, &rule(frame.top)), total));
    for (index, row) in rows.iter().enumerate() {
        let wrapped: Vec<Vec<String>> = row
            .iter()
            .zip(&widths)
            .map(|(cell, w)| wrap_text(cell, *w))
            .collect();
        let height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
        for line in 0..height {
            let cells: Vec<String> = wrapped
                .iter()
                .zip(&widths)
                .enumerate()
                .map(|(col, (lines, w))| {
                    let text = lines.get(line).map_or("", String::as_str);
                    let padded = pad(
                        text,
                        *w,
                        table.aligns.get(col).copied().unwrap_or(Align::Left),
                    );
                    if index == 0 {
                        theme.paint(&looks::META, &padded)
                    } else {
                        padded
                    }
                })
                .collect();
            out.push((
                format!("{bar} {} {bar}", cells.join(&format!(" {bar} "))),
                total,
            ));
        }
        if index == 0 {
            out.push((theme.paint(&looks::DIM, &rule(frame.middle)), total));
        }
    }
    out.push((theme.paint(&looks::DIM, &rule(frame.bottom)), total));
    Some(out)
}

/// The fallback for a table too wide for the terminal: one `header: value` block per row.
pub fn stacked(table: &Table, room: usize) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for (n, row) in table.rows.iter().enumerate() {
        if n > 0 {
            out.push((String::new(), 0));
        }
        for (name, value) in table.header.iter().zip(row) {
            let label = plain(name);
            let line = format!("{label}: {}", plain(value));
            for piece in wrap_text(&line, room) {
                let width = piece.width();
                out.push((piece, width));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(header: &[&str], rows: &[&[&str]]) -> Table {
        Table {
            header: header.iter().map(|s| s.to_string()).collect(),
            aligns: vec![Align::Left; header.len()],
            rows: rows
                .iter()
                .map(|r| r.iter().map(|s| s.to_string()).collect())
                .collect(),
        }
    }

    #[test]
    fn cells_trim_and_respect_escaped_bars() {
        assert_eq!(cells("| a | b |"), ["a", "b"]);
        assert_eq!(cells("a | b"), ["a", "b"]);
        assert_eq!(cells(r"| a\|b | c |"), ["a|b", "c"]);
        assert_eq!(cells("| a |  | c |"), ["a", "", "c"]);
    }

    #[test]
    fn delimiter_rows_give_alignments_and_everything_else_is_rejected() {
        assert_eq!(
            delimiter("|:---|:---:|---:|"),
            Some(vec![Align::Left, Align::Center, Align::Right])
        );
        assert_eq!(
            delimiter("| --- | --- |"),
            Some(vec![Align::Left, Align::Left])
        );
        assert_eq!(delimiter("| a | b |"), None);
        assert_eq!(delimiter("| - | -x- |"), None);
        assert_eq!(delimiter("||"), None);
    }

    #[test]
    fn a_table_that_fits_is_boxed_with_aligned_columns() {
        let t = table(&["name", "n"], &[&["alpha", "1"], &["b", "22"]]);
        let lines: Vec<String> = layout(&t, 40, &Theme::plain())
            .unwrap()
            .into_iter()
            .map(|l| l.0)
            .collect();
        assert_eq!(
            lines,
            [
                "+-------+----+",
                "| name  | n  |",
                "+-------+----+",
                "| alpha | 1  |",
                "| b     | 22 |",
                "+-------+----+",
            ]
        );
    }

    #[test]
    fn a_narrow_room_wraps_cells_and_never_exceeds_it() {
        let t = table(
            &["topic", "notes"],
            &[&["memory", "the quick brown fox jumps over the lazy dog"]],
        );
        for room in [30, 24, 20] {
            let lines = layout(&t, room, &Theme::plain()).unwrap();
            assert!(
                lines
                    .iter()
                    .all(|(text, w)| text.width() <= room && *w <= room),
                "{room}: {lines:?}"
            );
            assert!(lines.len() > 6, "the long cell wrapped over several lines");
        }
    }

    #[test]
    fn a_table_that_cannot_fit_at_all_falls_back_to_stacked_rows() {
        let t = table(
            &["a", "b", "c", "d", "e", "f"],
            &[&["1", "2", "3", "4", "5", "6"]],
        );
        assert!(layout(&t, 12, &Theme::plain()).is_none());
        let lines: Vec<String> = stacked(&t, 12).into_iter().map(|l| l.0).collect();
        assert_eq!(lines, ["a: 1", "b: 2", "c: 3", "d: 4", "e: 5", "f: 6"]);
    }

    #[test]
    fn alignment_and_inline_markers_in_cells() {
        let mut t = table(&["x"], &[&["**bold** `code` [link](http://u)"]]);
        t.aligns = vec![Align::Right];
        let lines = layout(&t, 60, &Theme::plain()).unwrap();
        assert!(
            lines.iter().any(|(l, _)| l.contains("bold code link")),
            "{lines:?}"
        );
        assert_eq!(wrap_text("", 5), [""]);
        assert_eq!(wrap_text("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    }
}
