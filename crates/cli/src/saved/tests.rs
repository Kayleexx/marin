use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use super::*;
use crate::code::Item;

pub(super) struct Temp(pub(super) PathBuf);

impl Temp {
    pub(super) fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("marin-saved-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    pub(super) fn store(&self) -> Store {
        Store::new(self.0.join("sessions"))
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) fn model() -> ModelId {
    ModelId {
        name: "tiny".into(),
        fingerprint: "00000000000000ff".into(),
    }
}

pub(super) fn pair(q: &str, a: &str) -> [Item; 2] {
    [Item::User(q.into()), Item::Assistant(a.into())]
}

pub(super) fn first(saved: Saved) -> Saved {
    Saved {
        revision: 1,
        ..saved
    }
}

pub(super) fn session(name: &str) -> Saved {
    let code = vec![
        Item::User("where is main?".into()),
        Item::Assistant("<tool_call>x</tool_call>".into()),
        Item::Results {
            done: vec![crate::code::Done {
                label: "read a.rs".into(),
                text: "fn main() {}".into(),
            }],
            note: String::new(),
            elided: false,
        },
        Item::Assistant("In a.rs:1.".into()),
    ];
    Saved::new(
        name.into(),
        (1_000, "code"),
        model(),
        pair("hi", "hello").to_vec(),
        Some(Code {
            root: PathBuf::from("/project"),
            items: code,
        }),
    )
}

/// Saves the way a conversation does: starting from the revision on disk, if there is one.
pub(super) fn put(store: &Store, saved: &Saved) -> Result<Written, SavedError> {
    let from = store.load(&saved.name).ok().map(|s| s.revision);
    store.save(saved, from)
}

pub(super) fn file(store: &Store, name: &str) -> PathBuf {
    store.dir().join(format!("{name}.json"))
}

#[test]
fn a_saved_session_comes_back_exactly_and_the_files_are_private() {
    let temp = Temp::new("roundtrip");
    let store = temp.store();
    let saved = session("work");
    assert!(put(&store, &saved).is_ok_and(|w| !w.replaced && w.revision == 1));
    assert_eq!(store.load("work").unwrap(), first(saved.clone()));
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(store.dir()), 0o700);
    assert_eq!(mode(&file(&store, "work")), 0o600);
    assert!(put(&store, &saved).is_ok_and(|w| w.replaced && w.revision == 2));
    assert_eq!(store.list().unwrap().len(), 1);
}

#[test]
fn a_write_that_died_leaves_the_committed_session_and_is_cleaned_up_later() {
    let temp = Temp::new("died");
    let store = temp.store();
    put(&store, &session("work")).unwrap();
    let half = store.dir().join(".123.456.tmp");
    fs::write(&half, b"{\"format\": \"marin-session\", \"vers").unwrap();
    assert_eq!(store.load("work").unwrap(), first(session("work")));
    assert_eq!(
        store.list().unwrap().len(),
        1,
        "a temporary file is not a session"
    );
    put(&store, &session("work")).unwrap();
    assert!(
        half.exists(),
        "a fresh temporary file may belong to a live run"
    );
    let old = SystemTime::now() - Duration::from_secs(3600);
    fs::File::options()
        .write(true)
        .open(&half)
        .unwrap()
        .set_modified(old)
        .unwrap();
    put(&store, &session("work")).unwrap();
    assert!(!half.exists());
}

#[test]
fn damaged_files_are_named_and_never_returned_partly() {
    let temp = Temp::new("damaged");
    let store = temp.store();
    put(&store, &session("good")).unwrap();
    let whole = fs::read(file(&store, "good")).unwrap();
    let cut = whole.get(..whole.len() / 2).unwrap();
    fs::write(file(&store, "cut"), cut).unwrap();
    fs::write(file(&store, "noise"), b"\xff\xfe not json").unwrap();
    fs::write(file(&store, "other"), b"{\"hello\": 1}").unwrap();
    let text = String::from_utf8(whole).unwrap();
    fs::write(
        file(&store, "future"),
        text.replace("\"version\":1", "\"version\":9"),
    )
    .unwrap();
    fs::write(
        file(&store, "old"),
        text.replace("\"version\":1", "\"version\":0"),
    )
    .unwrap();
    for name in ["cut", "noise", "other"] {
        assert!(
            matches!(store.load(name), Err(SavedError::Corrupt { .. })),
            "{name}"
        );
    }
    for name in ["future", "old"] {
        assert!(
            matches!(store.load(name), Err(SavedError::Version { .. })),
            "{name}"
        );
    }
    let listed = store.list().unwrap();
    assert_eq!(listed.len(), 6);
    assert!(matches!(listed.first(), Some(Listed::Ok(s)) if s.name == "good"));
    assert_eq!(store.load("good").unwrap(), first(session("good")));
}

#[test]
fn a_file_that_is_not_ours_is_never_replaced() {
    let temp = Temp::new("unrelated");
    let store = temp.store();
    put(&store, &session("keep")).unwrap();
    fs::write(file(&store, "notes"), b"my own notes").unwrap();
    assert!(matches!(
        put(&store, &session("notes")),
        Err(SavedError::NotOurs { .. })
    ));
    assert_eq!(fs::read(file(&store, "notes")).unwrap(), b"my own notes");
    let renamed = fs::read_to_string(file(&store, "keep")).unwrap();
    fs::write(file(&store, "liar"), renamed).unwrap();
    assert!(matches!(
        store.load("liar"),
        Err(SavedError::Corrupt { .. })
    ));
}

#[test]
fn a_session_larger_than_the_limit_is_refused_both_ways() {
    let temp = Temp::new("big");
    let store = temp.store();
    let mut big = session("big");
    big.chat = pair(&"x".repeat(usize::try_from(MAX_FILE).unwrap()), "ok").to_vec();
    assert!(matches!(put(&store, &big), Err(SavedError::TooBig { .. })));
    assert!(!file(&store, "big").exists());
    put(&store, &session("small")).unwrap();
    let huge = vec![b' '; usize::try_from(MAX_FILE).unwrap() + 10];
    fs::write(file(&store, "huge"), huge).unwrap();
    assert!(matches!(store.load("huge"), Err(SavedError::TooBig { .. })));
}

#[test]
fn only_plain_names_are_used_and_only_a_private_real_folder_is_trusted() {
    let temp = Temp::new("names");
    let store = temp.store();
    for bad in ["", "../x", "a/b", ".hidden", "-x", "a b", &"n".repeat(65)] {
        assert!(
            matches!(store.load(bad), Err(SavedError::BadName(_))),
            "{bad:?}"
        );
        assert!(put(&store, &session(bad)).is_err(), "{bad:?}");
    }
    put(&store, &session("fine.v1_x-2")).unwrap();
    fs::set_permissions(store.dir(), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(matches!(
        store.load("fine.v1_x-2"),
        Err(SavedError::UnsafeDir { .. })
    ));
    assert!(matches!(
        put(&store, &session("more")),
        Err(SavedError::UnsafeDir { .. })
    ));
    fs::set_permissions(store.dir(), fs::Permissions::from_mode(0o700)).unwrap();

    let elsewhere = temp.0.join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    fs::set_permissions(&elsewhere, fs::Permissions::from_mode(0o700)).unwrap();
    let linked = Store::new(temp.0.join("linked"));
    symlink(&elsewhere, linked.dir()).unwrap();
    assert!(matches!(
        linked.save(&session("x"), None),
        Err(SavedError::UnsafeDir { .. })
    ));
    assert!(
        fs::read_dir(&elsewhere).unwrap().next().is_none(),
        "nothing went through the link"
    );

    symlink(file(&store, "fine.v1_x-2"), file(&store, "alias")).unwrap();
    assert!(matches!(
        store.load("alias"),
        Err(SavedError::Corrupt { .. })
    ));
}

#[test]
fn the_number_of_sessions_is_bounded_but_a_known_one_can_always_be_updated() {
    let temp = Temp::new("many");
    let store = temp.store();
    for n in 0..MAX_SESSIONS {
        put(&store, &session(&format!("s{n}"))).unwrap();
    }
    assert!(matches!(
        put(&store, &session("one-more")),
        Err(SavedError::TooMany { .. })
    ));
    assert!(put(&store, &session("s7")).is_ok_and(|w| w.replaced));
}

#[test]
fn a_history_must_be_whole_exchanges_and_the_model_must_match() {
    let mut odd = session("odd");
    odd.chat.pop();
    assert!(odd.check().is_err(), "a question without its answer");
    let mut shuffled = session("shuffled");
    if let Some(code) = shuffled.code.as_mut() {
        code.items.remove(0);
    }
    assert!(
        shuffled.check().is_err(),
        "code history must start with a question"
    );
    let mut ends_early = session("early");
    if let Some(code) = ends_early.code.as_mut() {
        code.items.pop();
    }
    assert!(
        ends_early.check().is_err(),
        "an answer in progress is not a committed turn"
    );
    let mut unanswered = session("unanswered");
    if let Some(code) = unanswered.code.as_mut() {
        code.items.insert(3, Item::User("again".into()));
    }
    assert!(
        unanswered.check().is_err(),
        "tool output must be followed by an answer"
    );
    let mut lost = session("lost");
    lost.code = None;
    assert!(lost.check().is_err(), "code mode without a conversation");

    let saved = session("ok");
    assert!(saved.fits(&model()).is_ok());
    let other = ModelId {
        name: "big".into(),
        ..model()
    };
    assert!(matches!(
        saved.fits(&other),
        Err(SavedError::OtherModel { .. })
    ));
    let changed = ModelId {
        fingerprint: "1".into(),
        ..model()
    };
    assert!(matches!(
        saved.fits(&changed),
        Err(SavedError::ChangedModel { .. })
    ));
}

#[test]
fn listings_tell_sessions_apart() {
    let mut saved = session("work");
    saved.chat = pair("what is a mutex? and more\nsecond line", "a lock").to_vec();
    assert_eq!(saved.title(), "what is a mutex? and more");
    assert_eq!(saved.exchanges(), (1, 1));
    assert_eq!(ago(1_000, 990), "just now");
    assert_eq!(ago(10_000, 1_000), "2 h ago");
    assert_eq!(ago(1_000_000, 1_000), "11 d ago");
}
