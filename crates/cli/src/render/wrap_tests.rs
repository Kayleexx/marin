use super::*;

fn wrap(width: usize, margin: usize) -> Wrap {
    Wrap::new(Theme::plain(), width, margin, false)
}

fn say(w: &mut Wrap, text: &str) {
    for c in text.chars() {
        w.char(Look::NONE, c);
    }
}

fn prefix(text: &str) -> Prefix {
    Prefix {
        text: text.into(),
        width: text.chars().count(),
    }
}

#[test]
fn words_wrap_at_the_width_and_continuations_use_the_hanging_prefix() {
    let mut w = wrap(14, 0);
    w.begin(prefix("- "), prefix("  "));
    say(&mut w, "alpha beta gamma delta");
    w.end_line();
    assert_eq!(w.take(), "- alpha beta\n  gamma delta\n");
}

#[test]
fn the_margin_leads_every_line_but_the_open_first_one() {
    let mut w = Wrap::new(Theme::plain(), 20, 4, true);
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "one two three four five");
    w.end_line();
    assert_eq!(w.take(), "one two three\n    four five\n");
}

#[test]
fn a_word_longer_than_the_line_is_split_and_nothing_overflows() {
    let mut w = wrap(10, 0);
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "abcdefghijklmnopqrstuvwxyz");
    w.end_line();
    let text = w.take();
    assert!(text.lines().all(|l| l.chars().count() <= 10), "{text:?}");
    assert_eq!(text.replace('\n', ""), "abcdefghijklmnopqrstuvwxyz");
}

#[test]
fn wide_characters_count_two_columns() {
    let mut w = wrap(9, 0);
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "你好 世界 你好");
    w.end_line();
    assert_eq!(w.take(), "你好 世界\n你好\n");
    let mut narrow = wrap(8, 0);
    narrow.begin(Prefix::default(), Prefix::default());
    say(&mut narrow, "你好 世界 你好");
    narrow.end_line();
    assert_eq!(narrow.take(), "你好\n世界\n你好\n");
}

#[test]
fn blank_lines_collapse_and_never_lead_the_output() {
    let mut w = wrap(20, 0);
    w.blank_line();
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "a");
    w.blank_line();
    w.blank_line();
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "b");
    w.end_line();
    assert_eq!(w.take(), "a\n\nb\n");
}

#[test]
fn urls_are_underlined_only_when_color_is_on() {
    let color = Theme {
        color: true,
        unicode: true,
    };
    let mut w = Wrap::new(color, 40, 0, false);
    w.begin(Prefix::default(), Prefix::default());
    for c in "see https://x.dev now".chars() {
        w.char(Look::NONE, c);
    }
    w.end_line();
    assert_eq!(w.take(), "see \x1b[4;36mhttps://x.dev\x1b[0m now\n");
}

#[test]
fn an_empty_prefixed_line_keeps_its_gutter_without_trailing_space() {
    let mut w = wrap(20, 2);
    w.begin(prefix("| "), prefix("|   "));
    w.end_line_prefixed();
    assert_eq!(w.take(), "  |\n");
}

#[test]
fn an_open_first_line_stays_open_until_something_is_written_on_it() {
    let mut w = Wrap::new(Theme::plain(), 20, 4, true);
    w.begin(Prefix::default(), Prefix::default());
    w.end_line();
    w.begin(Prefix::default(), Prefix::default());
    say(&mut w, "hi");
    w.end_line();
    assert_eq!(
        w.take(),
        "hi\n",
        "no second margin on the line the caller already led in"
    );
}

#[test]
fn the_first_block_can_leave_the_prefix_line() {
    let mut w = Wrap::new(Theme::plain(), 20, 4, true);
    w.fresh_line();
    w.begin(prefix("| "), prefix("| "));
    w.put(Look::NONE, 'x');
    w.end_line();
    assert_eq!(w.take(), "\n    | x\n");
}
