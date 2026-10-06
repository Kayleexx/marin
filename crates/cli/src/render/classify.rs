/// What a source line turns out to be, decided from as few leading characters as possible.
/// `skip` is how many leading bytes (all ASCII) are syntax and not text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Blank,
    Paragraph {
        skip: usize,
    },
    Heading {
        level: usize,
        skip: usize,
    },
    Bullet {
        indent: usize,
        skip: usize,
    },
    Ordered {
        indent: usize,
        number: String,
        skip: usize,
    },
    Quote {
        skip: usize,
    },
    Rule,
    Fence {
        marker: char,
        len: usize,
        info: String,
    },
    TableRow,
}

fn paragraph(indent: usize) -> Option<Kind> {
    Some(Kind::Paragraph { skip: indent })
}

/// `None` means the line so far could still become several things: wait for more characters.
/// `complete` says the line has ended, which settles every case.
pub fn classify(line: &str, complete: bool) -> Option<Kind> {
    let rest = line.trim_start_matches(' ');
    let indent = line.len().saturating_sub(rest.len());
    let Some(first) = rest.chars().next() else {
        return complete.then_some(Kind::Blank);
    };
    match first {
        '#' => heading(rest, indent, complete),
        '`' | '~' => fence(rest, first, indent, complete),
        '>' => Some(Kind::Quote {
            skip: indent.saturating_add(1),
        }),
        '-' | '*' | '+' | '_' => marker(rest, first, indent, complete),
        '0'..='9' => ordered(rest, indent, complete),
        '|' => Some(Kind::TableRow),
        _ => paragraph(indent),
    }
}

fn heading(rest: &str, indent: usize, complete: bool) -> Option<Kind> {
    let level = rest.chars().take_while(|c| *c == '#').count();
    if level > 6 {
        return paragraph(indent);
    }
    match rest.chars().nth(level) {
        Some(' ') => Some(Kind::Heading {
            level,
            skip: indent.saturating_add(level).saturating_add(1),
        }),
        Some(_) => paragraph(indent),
        None if complete => Some(Kind::Blank),
        None => None,
    }
}

fn fence(rest: &str, marker: char, indent: usize, complete: bool) -> Option<Kind> {
    let len = rest.chars().take_while(|c| *c == marker).count();
    if len < 3 {
        return (rest.chars().count() > len || complete)
            .then_some(Kind::Paragraph { skip: indent });
    }
    if !complete {
        return None;
    }
    let info: String = rest
        .chars()
        .skip(len)
        .collect::<String>()
        .trim()
        .to_string();
    if marker == '`' && info.contains('`') {
        return paragraph(indent);
    }
    Some(Kind::Fence { marker, len, info })
}

fn marker(rest: &str, first: char, indent: usize, complete: bool) -> Option<Kind> {
    match rest.chars().nth(1) {
        None => complete.then_some(Kind::Paragraph { skip: indent }),
        Some(' ') if first != '_' => Some(Kind::Bullet {
            indent,
            skip: indent.saturating_add(2),
        }),
        Some(_) if rest.chars().all(|c| c == first || c == ' ') => {
            if !complete {
                None
            } else if rest.chars().filter(|c| *c == first).count() >= 3 {
                Some(Kind::Rule)
            } else {
                paragraph(indent)
            }
        }
        Some(_) => paragraph(indent),
    }
}

fn ordered(rest: &str, indent: usize, complete: bool) -> Option<Kind> {
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    if digits > 9 {
        return paragraph(indent);
    }
    let after: Vec<char> = rest.chars().skip(digits).take(2).collect();
    match after.as_slice() {
        [punct @ ('.' | ')'), ' '] => Some(Kind::Ordered {
            indent,
            number: format!("{}{punct}", rest.chars().take(digits).collect::<String>()),
            skip: indent.saturating_add(digits).saturating_add(2),
        }),
        [] | ['.' | ')'] if !complete => None,
        _ => paragraph(indent),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(line: &str) -> Option<Kind> {
        classify(line, true)
    }

    #[test]
    fn headings_need_a_space_after_the_hashes() {
        assert_eq!(kind("## Title"), Some(Kind::Heading { level: 2, skip: 3 }));
        assert_eq!(kind("#hashtag"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("####### seven"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(classify("##", false), None);
        assert_eq!(
            classify("## ", false),
            Some(Kind::Heading { level: 2, skip: 3 })
        );
    }

    #[test]
    fn bullets_and_rules_and_emphasis_share_markers() {
        assert_eq!(kind("- item"), Some(Kind::Bullet { indent: 0, skip: 2 }));
        assert_eq!(
            kind("   * nested"),
            Some(Kind::Bullet { indent: 3, skip: 5 })
        );
        assert_eq!(kind("---"), Some(Kind::Rule));
        assert_eq!(kind("* * *"), Some(Kind::Bullet { indent: 0, skip: 2 }));
        assert_eq!(kind("***"), Some(Kind::Rule));
        assert_eq!(kind("--"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("**bold** text"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("-5 degrees"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("_italic_"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(classify("**", false), None);
        assert_eq!(classify("---", false), None);
        assert_eq!(classify("-", false), None);
    }

    #[test]
    fn ordered_items_keep_their_number_and_ignore_plain_numbers() {
        assert_eq!(
            kind("12. twelve"),
            Some(Kind::Ordered {
                indent: 0,
                number: "12.".into(),
                skip: 4
            })
        );
        assert_eq!(kind("3.14 is pi"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("2024-01-01"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(
            kind("1) one"),
            Some(Kind::Ordered {
                indent: 0,
                number: "1)".into(),
                skip: 3
            })
        );
        assert_eq!(classify("12", false), None);
        assert_eq!(classify("12.", false), None);
    }

    #[test]
    fn fences_carry_their_info_string_and_wait_for_the_whole_line() {
        assert_eq!(classify("```rust", false), None);
        assert_eq!(
            kind("```rust"),
            Some(Kind::Fence {
                marker: '`',
                len: 3,
                info: "rust".into()
            })
        );
        assert_eq!(
            kind("~~~~"),
            Some(Kind::Fence {
                marker: '~',
                len: 4,
                info: String::new()
            })
        );
        assert_eq!(kind("```inline``` code"), Some(Kind::Paragraph { skip: 0 }));
        assert_eq!(kind("`code` span"), Some(Kind::Paragraph { skip: 0 }));
    }

    #[test]
    fn blank_quote_and_table_lines_are_recognized() {
        assert_eq!(kind(""), Some(Kind::Blank));
        assert_eq!(kind("   "), Some(Kind::Blank));
        assert_eq!(classify("   ", false), None);
        assert_eq!(kind("> quoted"), Some(Kind::Quote { skip: 1 }));
        assert_eq!(kind("| a | b |"), Some(Kind::TableRow));
        assert_eq!(kind("#"), Some(Kind::Blank));
    }
}
