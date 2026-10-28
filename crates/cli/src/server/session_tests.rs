use super::*;

fn hot(id: &str) -> Stored {
    Stored {
        id: id.to_owned(),
        cache: EngineCache::vacant().expect("vacant cache"),
        retained: Retained::default(),
    }
}

fn ids(store: &Store) -> Vec<String> {
    store.lock().cold.iter().map(|c| c.id.clone()).collect()
}

#[test]
fn a_claimed_session_cannot_be_claimed_again_until_the_lease_ends() {
    let store = Store::default();
    store.lock().hot = Some(hot("a"));
    let (lease, entry) = store.claim("a").expect("first claim");
    assert!(matches!(entry, Entry::Hot(_)));
    assert!(matches!(store.claim("a"), Err(Refused::Running)));
    assert!(store.lock().hot.is_none(), "the cache left the store");
    drop(lease);
    assert!(matches!(store.claim("a"), Err(Refused::Missing)));
}

#[test]
fn an_unknown_id_is_missing_not_running() {
    let store = Store::default();
    assert!(matches!(store.claim("nope"), Err(Refused::Missing)));
    assert!(store.lock().running.is_empty());
}

#[test]
fn a_cold_session_is_claimed_once_and_leaves_the_budget() {
    let store = Store::default();
    store.add_cold(Cold::placeholder("a", 100));
    store.add_cold(Cold::placeholder("b", 50));
    assert_eq!(store.lock().cold_bytes, 150);
    let (_lease, entry) = store.claim("a").expect("claim");
    assert!(matches!(entry, Entry::Cold(_)));
    assert_eq!(store.lock().cold_bytes, 50);
    assert_eq!(ids(&store), ["b"]);
}

#[test]
fn a_returned_cold_session_replaces_its_old_copy() {
    let store = Store::default();
    store.add_cold(Cold::placeholder("a", 100));
    store.add_cold(Cold::placeholder("b", 10));
    store.add_cold(Cold::placeholder("a", 30));
    assert_eq!(ids(&store), ["b", "a"]);
    assert_eq!(store.lock().cold_bytes, 40);
}

#[test]
fn the_oldest_cold_sessions_go_first_when_the_count_is_over() {
    let store = Store::default();
    for n in 0..COLD_SESSIONS + 3 {
        store.add_cold(Cold::placeholder(&format!("s{n}"), 1));
    }
    let kept = ids(&store);
    assert_eq!(kept.len(), COLD_SESSIONS);
    assert_eq!(kept.first().map(String::as_str), Some("s3"));
    assert!(matches!(store.claim("s0"), Err(Refused::Missing)));
}

#[test]
fn the_oldest_cold_sessions_go_first_when_the_bytes_are_over() {
    let store = Store::default();
    let third = COLD_BYTES / 3 + 1;
    for id in ["a", "b", "c"] {
        store.add_cold(Cold::placeholder(id, third));
    }
    assert_eq!(ids(&store), ["b", "c"]);
    assert!(store.lock().cold_bytes <= COLD_BYTES);
}

#[test]
fn a_session_larger_than_the_whole_budget_is_not_kept() {
    let store = Store::default();
    store.add_cold(Cold::placeholder("small", 10));
    store.add_cold(Cold::placeholder("huge", COLD_BYTES + 1));
    assert_eq!(ids(&store), ["small"]);
    assert_eq!(store.lock().cold_bytes, 10);
}

#[test]
fn leases_of_one_id_do_not_release_each_other() {
    let store = Store::default();
    store.add_cold(Cold::placeholder("a", 1));
    let (first, _) = store.claim("a").expect("claim");
    drop(first);
    store.add_cold(Cold::placeholder("a", 1));
    let (second, _) = store.claim("a").expect("claim again");
    assert_eq!(store.lock().running.len(), 1);
    drop(second);
    assert!(store.lock().running.is_empty());
}

#[test]
fn a_session_being_spilled_is_waited_for_not_reported_missing() {
    let store = Store::default();
    store.lock().hot = Some(hot("a"));
    let (old, moving) = store.displace(None).expect("hot");
    let (sent, got) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            sent.send(store.claim("a").map(|(_, entry)| entry))
                .expect("receiver")
        });
        assert!(
            got.recv_timeout(std::time::Duration::from_millis(300))
                .is_err(),
            "the claim must wait while the session is moving to host memory"
        );
        store.add_cold(Cold::placeholder(&old.id, 10));
        drop(moving);
        let claimed = got.recv().expect("claim answered");
        assert!(
            matches!(claimed, Ok(Entry::Cold(_))),
            "the claim finds the cold copy"
        );
    });
}
