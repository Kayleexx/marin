use std::io::{self, Read};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub(super) fn watch_disconnect(
    stream: &TcpStream,
    cancel: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    timed_out: Arc<AtomicBool>,
    limit: Duration,
) -> io::Result<thread::JoinHandle<()>> {
    let mut peer = stream.try_clone()?;
    peer.set_read_timeout(Some(Duration::from_millis(100)))?;
    Ok(thread::spawn(move || {
        let mut byte = [0_u8; 1];
        let start = Instant::now();
        let mut half_closed = false;
        while !done.load(Ordering::Relaxed) {
            if start.elapsed() >= limit {
                timed_out.store(true, Ordering::Relaxed);
                cancel.store(true, Ordering::Relaxed);
                break;
            }
            if half_closed {
                thread::sleep(Duration::from_millis(100));
                continue;
            }
            match peer.read(&mut byte) {
                Ok(0) => half_closed = true,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) => {}
                Ok(_) | Err(_) => {
                    if !done.load(Ordering::Relaxed) {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    break;
                }
            }
        }
    }))
}
