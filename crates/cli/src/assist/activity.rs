use serde::{Deserialize, Serialize};

use super::redact::redact;

pub const MAX_CONTEXT: usize = 32;
pub const CONTEXT_TTL: u64 = 30 * 60;
pub const SHARE_BYTES: usize = 4096;
const TITLE_CHARS: usize = 160;
const PATH_CHARS: usize = 120;
const COMMON: [&str; 12] = [
    "error", "failed", "panicked", "fail", "expected", "found", "cannot", "could", "with", "from",
    "that", "this",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Window,
    Tab,
    Share,
}

impl Source {
    pub fn name(self) -> &'static str {
        match self {
            Source::Window => "window",
            Source::Tab => "tab",
            Source::Share => "shared",
        }
    }
}

/// What the user was looking at: the application or page and its title, never its content (except text the user
/// shared by hand). Addresses keep only the host and path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    pub source: Source,
    #[serde(default)]
    pub app: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub at: u64,
}

fn cut(text: &str, chars: usize) -> String {
    text.chars()
        .filter(|c| !c.is_control())
        .take(chars)
        .collect()
}

/// `https://user@Host/path?q#f` becomes `("host", "host/path")`. Only http and https addresses are accepted.
pub fn host_and_path(url: &str) -> Option<(String, String)> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let rest = rest.split(['?', '#']).next().unwrap_or("");
    let (authority, path) = rest.split_once('/').unwrap_or((rest, ""));
    let host = authority.rsplit('@').next()?.split(':').next()?;
    let host = host.to_ascii_lowercase();
    (!host.is_empty()
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-._".contains(c)))
    .then(|| (host.clone(), cut(&format!("{host}/{path}"), PATH_CHARS)))
}

impl Activity {
    /// Filters and bounds what an adapter sent; `None` when there is nothing usable.
    pub fn cleaned(mut self, now: u64) -> Option<Self> {
        self.at = now;
        self.app = cut(&self.app, 60).to_ascii_lowercase();
        self.title = cut(&redact(&self.title), TITLE_CHARS);
        if !self.url.is_empty() {
            self.url = host_and_path(&self.url)?.1;
        }
        let text = redact(&self.text);
        self.text = if self.source == Source::Share {
            super::inbox::bound(&text, SHARE_BYTES)
        } else {
            String::new()
        };
        (!(self.title.is_empty() && self.url.is_empty() && self.text.is_empty())).then_some(self)
    }

    pub fn host(&self) -> &str {
        self.url.split('/').next().unwrap_or("")
    }

    pub fn line(&self, now: u64) -> String {
        let place = match (self.app.is_empty(), self.url.is_empty()) {
            (false, _) => self.app.clone(),
            (true, false) => self.url.clone(),
            _ => String::new(),
        };
        format!(
            "{:>4} s ago  {:<6} {place}  {}",
            now.saturating_sub(self.at),
            self.source.name(),
            self.title
        )
    }
}

fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(str::to_ascii_lowercase)
        .filter(|w| !COMMON.contains(&w.as_str()))
        .collect()
}

/// Does the page look like it is about this error? The page's title or address must share a distinctive word
/// (an error code, a crate or a message word) with the error line.
pub fn related(error_line: &str, activity: &Activity) -> bool {
    let wanted = words(error_line);
    words(&format!("{} {}", activity.title, activity.url))
        .iter()
        .any(|w| wanted.contains(w))
}

/// The recent activity, newest last, bounded in count and age.
#[derive(Default)]
pub struct Context {
    items: Vec<Activity>,
}

impl Context {
    pub fn push(&mut self, activity: Activity) {
        self.expire(activity.at);
        if self.items.last().is_some_and(|l| {
            l.source == activity.source
                && l.app == activity.app
                && l.title == activity.title
                && l.url == activity.url
        }) {
            return;
        }
        self.items.push(activity);
        let extra = self.items.len().saturating_sub(MAX_CONTEXT);
        self.items.drain(..extra);
    }

    pub fn expire(&mut self, now: u64) {
        self.items
            .retain(|a| now.saturating_sub(a.at) < CONTEXT_TTL);
    }

    pub fn items(&self) -> &[Activity] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(title: &str, url: &str) -> Activity {
        Activity {
            source: Source::Tab,
            app: String::new(),
            title: title.into(),
            url: url.into(),
            text: String::new(),
            at: 0,
        }
    }

    #[test]
    fn addresses_lose_queries_credentials_and_odd_schemes() {
        let url = |u: &str| host_and_path(u).map(|p| p.1);
        assert_eq!(
            url("https://me:pw@Docs.RS/tokio/latest?token=abc#x").as_deref(),
            Some("docs.rs/tokio/latest")
        );
        assert_eq!(url("file:///etc/passwd"), None);
        assert_eq!(url("https:///x"), None);
        assert!(tab("t", "javascript:alert(1)").cleaned(0).is_none());
    }

    #[test]
    fn a_page_is_related_by_a_distinctive_word_not_by_a_generic_one() {
        let error = "error[E0382]: borrow of moved value: `config`";
        let page = tab(
            "E0382 - Rust Compiler Error Index",
            "https://doc.rust-lang.org/error_codes/E0382.html",
        );
        assert!(related(error, &page.cleaned(0).unwrap()));
        let other = tab("Error handling in Go", "https://go.dev/blog/error");
        assert!(!related(error, &other.cleaned(0).unwrap()));
    }

    #[test]
    fn the_context_is_bounded_deduplicated_and_expires() {
        let mut ctx = Context::default();
        for n in 0..50 {
            let mut a = tab(&format!("page {n}"), "https://a.dev/x")
                .cleaned(n)
                .unwrap();
            a.at = n;
            ctx.push(a);
        }
        assert_eq!(ctx.items().len(), MAX_CONTEXT);
        let again = ctx.items().last().cloned().unwrap();
        ctx.push(again);
        assert_eq!(ctx.items().len(), MAX_CONTEXT);
        ctx.expire(49 + CONTEXT_TTL + 1);
        assert!(ctx.items().is_empty());
    }

    #[test]
    fn shared_text_is_filtered_and_bounded_but_tab_text_is_dropped() {
        let mut shared = tab("notes", "");
        shared.source = Source::Share;
        shared.text = format!("password=hunter2 {}", "x".repeat(10_000));
        let shared = shared.cleaned(0).unwrap();
        assert!(shared.text.len() <= SHARE_BYTES && !shared.text.contains("hunter2"));
        let mut t = tab("a", "https://a.dev");
        t.text = "page body".into();
        assert!(t.cleaned(0).unwrap().text.is_empty());
    }
}
