//! Two runs saving one session, and folders or paths that someone else controls.

use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::time::Duration;

use super::tests::{Temp, file, session};
use super::*;

#[test]
fn a_save_that_did_not_start_from_the_current_revision_is_a_conflict_not_a_lost_update() {
    let temp = Temp::new("conflict");
    let store = temp.store();
    let first = store.save(&session("work"), None).unwrap();
    assert_eq!(first.revision, 1);
    let other = store.save(&session("work"), Some(1)).unwrap();
    assert_eq!(other.revision, 2, "the other run saved from revision 1");
    let late = store.save(&session("work"), Some(1));
    assert!(
        matches!(
            late,
            Err(SavedError::Conflict {
                found: 2,
                seen: Some(1),
                ..
            })
        ),
        "{late:?}"
    );
    let stranger = store.save(&session("work"), None);
    assert!(
        matches!(stranger, Err(SavedError::Conflict { seen: None, .. })),
        "{stranger:?}"
    );
    assert_eq!(store.load("work").unwrap().revision, 2);
    let mut old = serde_json::to_value(session("legacy")).unwrap();
    old.as_object_mut().unwrap().remove("revision");
    fs::create_dir_all(store.dir()).unwrap();
    fs::write(file(&store, "legacy"), old.to_string()).unwrap();
    assert_eq!(
        store.load("legacy").unwrap().revision,
        0,
        "files from before revisions still load"
    );
    assert_eq!(store.save(&session("legacy"), Some(0)).unwrap().revision, 1);
}

#[test]
fn saves_racing_for_one_session_never_lose_an_update() {
    let temp = Temp::new("race");
    let store = temp.store();
    store.save(&session("shared"), None).unwrap();
    let wins = std::sync::atomic::AtomicUsize::new(0);
    let conflicts = std::sync::atomic::AtomicUsize::new(0);
    let order = std::sync::atomic::Ordering::SeqCst;
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                let racer = Store::new(store.dir().to_path_buf());
                match racer.save(&session("shared"), Some(1)) {
                    Ok(_) => wins.fetch_add(1, order),
                    Err(SavedError::Conflict { .. }) => conflicts.fetch_add(1, order),
                    Err(other) => panic!("{other}"),
                };
            });
        }
    });
    assert_eq!(
        (wins.load(order), conflicts.load(order)),
        (1, 7),
        "exactly one run wins"
    );
    assert_eq!(store.load("shared").unwrap().revision, 2);
}

#[test]
fn a_save_waits_for_a_run_that_is_saving_and_gives_up_clearly() {
    let temp = Temp::new("busy");
    let store = temp.store();
    store.save(&session("work"), None).unwrap();
    let lock = fs::File::options()
        .write(true)
        .open(store.dir().join(".lock"))
        .unwrap();
    lock.lock().unwrap();
    let began = std::time::Instant::now();
    let refused = store.save(&session("work"), Some(1));
    assert!(
        matches!(refused, Err(SavedError::Busy { .. })),
        "{refused:?}"
    );
    assert!(began.elapsed() < Duration::from_secs(10));
    drop(lock);
    assert!(
        store.save(&session("work"), Some(1)).is_ok(),
        "the lock goes with its holder"
    );
    let link = store.dir().join(".lock");
    fs::remove_file(&link).unwrap();
    symlink(temp.0.join("elsewhere"), &link).unwrap();
    assert!(matches!(
        store.save(&session("work"), Some(2)),
        Err(SavedError::UnsafeDir { .. })
    ));
}

fn mine() -> u32 {
    use std::os::unix::fs::MetadataExt;
    fs::metadata("/proc/self").unwrap().uid()
}

#[test]
fn a_folder_or_path_that_someone_else_controls_is_not_trusted() {
    let temp = Temp::new("owners");
    let ours = temp.0.join("sessions");
    let real = Store::new(ours.clone());
    real.save(&session("work"), None).unwrap();

    let stranger = Store::trusting(ours.clone(), Trust::as_user(mine().wrapping_add(1)));
    let seen = stranger.load("work");
    assert!(
        matches!(seen, Err(SavedError::UnsafeDir { .. })),
        "{seen:?}"
    );
    assert!(stranger.save(&session("work"), Some(1)).is_err());

    let shared = temp.0.join("shared");
    fs::create_dir(&shared).unwrap();
    fs::set_permissions(&shared, fs::Permissions::from_mode(0o777)).unwrap();
    let under = Store::new(shared.join("sessions"));
    assert!(matches!(
        under.save(&session("work"), None),
        Err(SavedError::UnsafeDir { .. })
    ));
    assert!(
        !shared.join("sessions").exists(),
        "nothing was made under a folder anyone can change"
    );
    fs::set_permissions(&shared, fs::Permissions::from_mode(0o1777)).unwrap();
    assert!(
        under.save(&session("work"), None).is_ok(),
        "a sticky folder like /tmp is fine"
    );

    let real_parent = temp.0.join("real-parent");
    fs::create_dir(&real_parent).unwrap();
    symlink(&real_parent, temp.0.join("via-link")).unwrap();
    let through = Store::new(temp.0.join("via-link").join("sessions"));
    assert!(
        through.save(&session("work"), None).is_ok(),
        "a link of our own is followed"
    );
    let distrusting = Store::trusting(
        temp.0.join("via-link").join("sessions"),
        Trust::as_user(mine().wrapping_add(1)),
    );
    assert!(matches!(
        distrusting.load("work"),
        Err(SavedError::UnsafeDir { .. })
    ));
}
