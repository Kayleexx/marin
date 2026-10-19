use std::io;
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use super::watch::watch_disconnect;

#[test]
fn write_half_close_does_not_cancel_but_generation_deadline_does() -> io::Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let address = listener.local_addr()?;
    let client = TcpStream::connect(address)?;
    let (server, _) = listener.accept()?;
    client.shutdown(Shutdown::Write)?;
    let cancel = Arc::new(AtomicBool::new(false));
    let done = Arc::new(AtomicBool::new(false));
    let timed_out = Arc::new(AtomicBool::new(false));
    let watcher = watch_disconnect(
        &server,
        Arc::clone(&cancel),
        done,
        Arc::clone(&timed_out),
        Duration::from_millis(250),
    )?;
    thread::sleep(Duration::from_millis(150));
    assert!(!cancel.load(Ordering::Relaxed));
    let _ = watcher.join();
    assert!(timed_out.load(Ordering::Relaxed));
    assert!(cancel.load(Ordering::Relaxed));
    Ok(())
}
