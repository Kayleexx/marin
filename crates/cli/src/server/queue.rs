use std::io::{self, Read};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use super::send_error;

pub(super) const WAIT_LIMIT: Duration = Duration::from_secs(10);
pub(super) const READ_LIMIT: Duration = Duration::from_secs(10);
pub(super) const RUN_LIMIT: Duration = Duration::from_secs(300);

pub(super) fn expired(accepted: Instant) -> bool {
    accepted.elapsed() >= WAIT_LIMIT
}

pub(super) struct Pending {
    pub stream: TcpStream,
    pub id: u64,
    pub accepted: Instant,
}

pub(super) fn drain(stream: &mut TcpStream) {
    let start = Instant::now();
    let mut chunk = [0_u8; 4096];
    let mut remaining = 1024 * 1024 + 16 * 1024;
    while remaining > 0 {
        let Some(time) = Duration::from_millis(100)
            .checked_sub(start.elapsed())
            .filter(|time| !time.is_zero())
        else {
            break;
        };
        if stream.set_read_timeout(Some(time)).is_err() {
            break;
        }
        let amount = remaining.min(chunk.len());
        let Some(buf) = chunk.get_mut(..amount) else {
            break;
        };
        match stream.read(buf) {
            Ok(0) | Err(_) => break,
            Ok(count) => remaining = remaining.saturating_sub(count),
        }
    }
}

pub(super) fn listen(port: u16) -> io::Result<(SocketAddr, Receiver<Pending>)> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let address = listener.local_addr()?;
    let (sender, receiver) = mpsc::sync_channel(2);
    thread::Builder::new()
        .name("marin-accept".into())
        .spawn(move || {
            for (id, incoming) in listener.incoming().enumerate() {
                let stream = match incoming {
                    Ok(stream) => stream,
                    Err(error) => {
                        eprintln!("accept failed: {error}");
                        break;
                    }
                };
                let pending = Pending {
                    stream,
                    id: u64::try_from(id).unwrap_or(u64::MAX),
                    accepted: Instant::now(),
                };
                match sender.try_send(pending) {
                    Ok(()) => {}
                    Err(TrySendError::Full(mut pending)) => {
                        drain(&mut pending.stream);
                        let _ = pending
                            .stream
                            .set_write_timeout(Some(Duration::from_millis(100)));
                        let _ = send_error(&mut pending.stream, 503, "server queue is full");
                    }
                    Err(TrySendError::Disconnected(_)) => break,
                }
            }
        })?;
    Ok((address, receiver))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_wait_boundary() {
        assert!(!expired(Instant::now() - Duration::from_secs(5)));
        assert!(expired(
            Instant::now() - WAIT_LIMIT - Duration::from_millis(1)
        ));
    }
}
