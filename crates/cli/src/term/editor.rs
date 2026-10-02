use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

const MAX_HISTORY: usize = 500;

/// Single prompt line (with optional embedded newlines) and its history.
#[derive(Default)]
pub struct Editor {
    text: Vec<char>,
    cursor: usize,
    history: Vec<String>,
    position: Option<usize>,
    draft: Vec<char>,
}

impl Editor {
    pub fn with_history(history: Vec<String>) -> Self {
        Self {
            history,
            ..Self::default()
        }
    }

    pub fn with_text(text: &str) -> Self {
        let mut editor = Self::default();
        editor.set(text);
        editor
    }

    pub fn text(&self) -> String {
        self.text.iter().collect()
    }

    pub fn chars(&self) -> &[char] {
        &self.text
    }

    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn history(&self) -> &[String] {
        &self.history
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.position = None;
    }

    pub fn set(&mut self, text: &str) {
        self.text = text.chars().collect();
        self.cursor = self.text.len();
    }

    pub fn remember(&mut self, entry: &str) {
        if self.history.last().map(String::as_str) != Some(entry) {
            self.history.push(entry.to_string());
        }
        let excess = self.history.len().saturating_sub(MAX_HISTORY);
        self.history.drain(..excess);
    }

    pub fn insert(&mut self, text: &str) {
        let cleaned = text.replace("\r\n", "\n").replace('\r', "\n");
        for ch in cleaned.chars() {
            self.text.insert(self.cursor.min(self.text.len()), ch);
            self.cursor = self.cursor.saturating_add(1);
        }
    }

    fn delete_range(&mut self, from: usize, to: usize) {
        let to = to.min(self.text.len());
        if from < to {
            self.text.drain(from..to);
            self.cursor = from;
        }
    }

    fn word_start(&self) -> usize {
        let mut i = self.cursor;
        while i > 0
            && self
                .text
                .get(i.saturating_sub(1))
                .is_some_and(|c| c.is_whitespace())
        {
            i = i.saturating_sub(1);
        }
        while i > 0
            && self
                .text
                .get(i.saturating_sub(1))
                .is_some_and(|c| !c.is_whitespace())
        {
            i = i.saturating_sub(1);
        }
        i
    }

    fn recall(&mut self, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let last = self.history.len().saturating_sub(1);
        let next = match (self.position, older) {
            (None, true) => {
                self.draft = self.text.clone();
                Some(last)
            }
            (None, false) => return,
            (Some(p), true) => Some(p.saturating_sub(1)),
            (Some(p), false) if p < last => Some(p.saturating_add(1)),
            (Some(_), false) => None,
        };
        self.position = next;
        self.text = match next.and_then(|p| self.history.get(p)) {
            Some(entry) => entry.chars().collect(),
            None => std::mem::take(&mut self.draft),
        };
        self.cursor = self.text.len();
    }

    /// Applies an editing key. Returns false for keys the editor does not own.
    pub fn handle(&mut self, key: KeyEvent) -> bool {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let alt = key.modifiers.contains(KeyModifiers::ALT);
        let len = self.text.len();
        match key.code {
            KeyCode::Char('j') if ctrl => self.insert("\n"),
            KeyCode::Enter if alt => self.insert("\n"),
            KeyCode::Char('a') if ctrl => self.cursor = 0,
            KeyCode::Char('e') if ctrl => self.cursor = len,
            KeyCode::Char('u') if ctrl => self.delete_range(0, self.cursor),
            KeyCode::Char('k') if ctrl => self.delete_range(self.cursor, len),
            KeyCode::Char('w') if ctrl => self.delete_range(self.word_start(), self.cursor),
            KeyCode::Backspace if alt => self.delete_range(self.word_start(), self.cursor),
            KeyCode::Char(c) if !ctrl && !alt => self.insert(&c.to_string()),
            KeyCode::Backspace => self.delete_range(self.cursor.saturating_sub(1), self.cursor),
            KeyCode::Delete => self.delete_range(self.cursor, self.cursor.saturating_add(1)),
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = self.cursor.saturating_add(1).min(len),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = len,
            KeyCode::Up => self.recall(true),
            KeyCode::Down => self.recall(false),
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    fn typed(editor: &mut Editor, text: &str) {
        for c in text.chars() {
            assert!(editor.handle(key(KeyCode::Char(c))));
        }
    }

    #[test]
    fn typing_moving_and_deleting_edit_in_place() {
        let mut e = Editor::default();
        typed(&mut e, "helo");
        e.handle(key(KeyCode::Left));
        typed(&mut e, "l");
        assert_eq!(e.text(), "hello");
        e.handle(key(KeyCode::Backspace));
        assert_eq!(e.text(), "helo");
        e.handle(key(KeyCode::Home));
        e.handle(key(KeyCode::Delete));
        assert_eq!(e.text(), "elo");
        assert_eq!(e.cursor(), 0);
        e.handle(key(KeyCode::Left));
        e.handle(key(KeyCode::End));
        e.handle(key(KeyCode::Right));
        assert_eq!(e.cursor(), 3);
    }

    #[test]
    fn kill_and_word_shortcuts() {
        let mut e = Editor::default();
        typed(&mut e, "one two  three");
        e.handle(ctrl('w'));
        assert_eq!(e.text(), "one two  ");
        e.handle(ctrl('w'));
        assert_eq!(e.text(), "one ");
        e.handle(ctrl('a'));
        typed(&mut e, "x");
        e.handle(ctrl('k'));
        assert_eq!(e.text(), "x");
        e.handle(ctrl('u'));
        assert!(e.is_empty());
    }

    #[test]
    fn newlines_come_from_alt_enter_ctrl_j_and_paste() {
        let mut e = Editor::default();
        typed(&mut e, "a");
        e.handle(KeyEvent::new(KeyCode::Enter, KeyModifiers::ALT));
        e.handle(ctrl('j'));
        e.insert("b\r\nc");
        assert_eq!(e.text(), "a\n\nb\nc");
        assert!(
            !e.handle(key(KeyCode::Enter)),
            "plain Enter belongs to the app"
        );
        assert!(!e.handle(key(KeyCode::Esc)));
    }

    #[test]
    fn history_walks_back_and_restores_the_draft() {
        let mut e = Editor::with_history(vec!["first".into(), "second".into()]);
        typed(&mut e, "draft");
        e.handle(key(KeyCode::Up));
        assert_eq!(e.text(), "second");
        e.handle(key(KeyCode::Up));
        e.handle(key(KeyCode::Up));
        assert_eq!(e.text(), "first");
        e.handle(key(KeyCode::Down));
        assert_eq!(e.text(), "second");
        e.handle(key(KeyCode::Down));
        assert_eq!(e.text(), "draft");
        e.handle(key(KeyCode::Down));
        assert_eq!(e.text(), "draft");
    }

    #[test]
    fn remember_skips_repeats_and_caps_the_list() {
        let mut e = Editor::default();
        e.remember("a");
        e.remember("a");
        assert_eq!(e.history(), ["a"]);
        for i in 0..MAX_HISTORY + 5 {
            e.remember(&i.to_string());
        }
        assert_eq!(e.history().len(), MAX_HISTORY);
        let mut empty = Editor::default();
        empty.handle(key(KeyCode::Up));
        assert!(empty.is_empty());
    }
}
