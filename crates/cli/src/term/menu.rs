/// A list that narrows as the user types: the action palette and pick lists.
pub struct Menu<T> {
    items: Vec<Item<T>>,
    query: String,
    selected: usize,
}

pub struct Item<T> {
    pub label: String,
    pub detail: String,
    pub value: T,
}

pub struct Row<'a> {
    pub label: &'a str,
    pub detail: &'a str,
    pub selected: bool,
}

fn is_subsequence(text: &str, word: &str) -> bool {
    let mut letters = text.chars();
    word.chars().all(|w| letters.any(|c| c == w))
}

/// Lower is better; None means the word does not match.
fn word_score(label: &str, detail: &str, word: &str) -> Option<u32> {
    if label.starts_with(word) {
        Some(0)
    } else if label.contains(word) {
        Some(1)
    } else if is_subsequence(label, word) {
        Some(2)
    } else if detail.contains(word) {
        Some(3)
    } else {
        None
    }
}

impl<T> Menu<T> {
    pub fn new(items: Vec<Item<T>>) -> Self {
        Self {
            items,
            query: String::new(),
            selected: 0,
        }
    }

    pub fn query(&self) -> &str {
        &self.query
    }

    fn visible(&self) -> Vec<usize> {
        let query = self.query.to_lowercase();
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut scored: Vec<(u32, usize)> = self
            .items
            .iter()
            .enumerate()
            .filter_map(|(index, item)| {
                let (label, detail) = (item.label.to_lowercase(), item.detail.to_lowercase());
                words
                    .iter()
                    .try_fold(0_u32, |sum, w| {
                        word_score(&label, &detail, w).map(|s| sum.saturating_add(s))
                    })
                    .map(|score| (score, index))
            })
            .collect();
        scored.sort_by_key(|(score, index)| (*score, *index));
        scored.into_iter().map(|(_, index)| index).collect()
    }

    pub fn push(&mut self, c: char) {
        self.query.push(c);
        self.selected = 0;
    }

    /// Returns false when the query was already empty.
    pub fn pop(&mut self) -> bool {
        self.selected = 0;
        self.query.pop().is_some()
    }

    pub fn up(&mut self) {
        let count = self.visible().len();
        self.selected = match self.selected {
            0 => count.saturating_sub(1),
            n => n.saturating_sub(1),
        };
    }

    pub fn down(&mut self) {
        let count = self.visible().len();
        self.selected = if self.selected.saturating_add(1) >= count {
            0
        } else {
            self.selected.saturating_add(1)
        };
    }

    pub fn current(&self) -> Option<&T> {
        let index = *self.visible().get(self.selected)?;
        self.items.get(index).map(|item| &item.value)
    }

    /// Consumes the menu and returns the selected value.
    pub fn into_current(self) -> Option<T> {
        let index = *self.visible().get(self.selected)?;
        self.items.into_iter().nth(index).map(|item| item.value)
    }

    /// The first `max` matching rows, with the selected one flagged.
    pub fn rows(&self, max: usize) -> Vec<Row<'_>> {
        let visible = self.visible();
        let first = self.selected.saturating_add(1).saturating_sub(max);
        visible
            .iter()
            .enumerate()
            .skip(first)
            .take(max)
            .filter_map(|(position, index)| {
                self.items.get(*index).map(|item| Row {
                    label: &item.label,
                    detail: &item.detail,
                    selected: position == self.selected,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu() -> Menu<u32> {
        let item = |label: &str, detail: &str, value| Item {
            label: label.into(),
            detail: detail.into(),
            value,
        };
        Menu::new(vec![
            item("Inspect last reply", "metrics, kv and model", 1),
            item("Show tokens…", "how text is split", 2),
            item("Temperature…", "randomness", 3),
            item("Max tokens…", "reply length", 4),
        ])
    }

    fn type_query(menu: &mut Menu<u32>, text: &str) {
        text.chars().for_each(|c| menu.push(c));
    }

    #[test]
    fn empty_query_lists_everything_in_order() {
        let m = menu();
        assert_eq!(m.visible().len(), 4);
        assert_eq!(m.current(), Some(&1));
    }

    #[test]
    fn label_prefix_beats_contains_beats_detail() {
        let mut m = menu();
        type_query(&mut m, "tok");
        let labels: Vec<&str> = m.rows(10).iter().map(|r| r.label).collect();
        assert_eq!(labels, ["Show tokens…", "Max tokens…"]);
        let mut m = menu();
        type_query(&mut m, "temp");
        assert_eq!(m.current(), Some(&3));
        let mut m = menu();
        type_query(&mut m, "random");
        assert_eq!(m.current(), Some(&3), "matches the detail text");
    }

    #[test]
    fn several_words_must_all_match_and_fuzzy_letters_work() {
        let mut m = menu();
        type_query(&mut m, "last kv");
        assert_eq!(m.current(), Some(&1));
        let mut m = menu();
        type_query(&mut m, "mxtk");
        assert_eq!(m.current(), Some(&4));
        let mut m = menu();
        type_query(&mut m, "zzz");
        assert_eq!(m.visible().len(), 0);
        assert_eq!(m.current(), None);
    }

    #[test]
    fn selection_wraps_and_resets_when_the_query_changes() {
        let mut m = menu();
        m.up();
        assert_eq!(m.current(), Some(&4));
        m.down();
        assert_eq!(m.current(), Some(&1));
        m.down();
        m.down();
        assert_eq!(m.current(), Some(&3));
        m.push('t');
        assert_eq!(m.rows(10).first().map(|r| r.selected), Some(true));
        assert!(m.pop());
        assert!(!m.pop());
    }

    #[test]
    fn rows_scroll_to_keep_the_selection_visible() {
        let mut m = menu();
        m.up();
        let rows = m.rows(2);
        assert_eq!(rows.len(), 2);
        assert!(rows.last().is_some_and(|r| r.selected));
    }
}
