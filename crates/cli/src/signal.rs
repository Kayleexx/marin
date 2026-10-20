use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const LOADING: u8 = 0;
const GENERATING: u8 = 1;
const WAITING_FOR_INPUT: u8 = 2;

pub const INTERRUPTED_EXIT_CODE: i32 = 130;

/// Ctrl-C meaning depends on what the program is doing; see `install`.
#[derive(Clone)]
pub struct Interrupt {
    phase: Arc<AtomicU8>,
    armed: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
}

impl Interrupt {
    /// While loading: exit. While generating: stop the reply (a second press exits,
    /// since the first one only takes effect between tokens). While waiting for
    /// input: warn first, exit on the second press.
    pub fn install() -> Result<Self, ctrlc::Error> {
        let this = Self {
            phase: Arc::new(AtomicU8::new(LOADING)),
            armed: Arc::new(AtomicBool::new(false)),
            cancel: Arc::new(AtomicBool::new(false)),
        };
        let handler = this.clone();
        ctrlc::set_handler(move || handler.pressed())?;
        Ok(this)
    }

    /// An interrupt no key press reaches, for callers that only need a cancel flag they can set themselves.
    pub fn unhooked() -> Self {
        Self {
            phase: Arc::new(AtomicU8::new(LOADING)),
            armed: Arc::new(AtomicBool::new(false)),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    fn pressed(&self) {
        match self.phase.load(Ordering::SeqCst) {
            GENERATING if !self.cancel.swap(true, Ordering::SeqCst) => {}
            WAITING_FOR_INPUT if !self.armed.swap(true, Ordering::SeqCst) => {
                eprintln!("\n(press Ctrl-C again or type /quit to exit)");
            }
            _ => {
                eprintln!();
                process::exit(INTERRUPTED_EXIT_CODE);
            }
        }
    }

    pub fn cancel_flag(&self) -> &AtomicBool {
        &self.cancel
    }

    pub fn generating(&self) {
        self.cancel.store(false, Ordering::SeqCst);
        self.phase.store(GENERATING, Ordering::SeqCst);
    }

    pub fn waiting_for_input(&self) {
        self.armed.store(false, Ordering::SeqCst);
        self.phase.store(WAITING_FOR_INPUT, Ordering::SeqCst);
    }

    pub fn busy(&self) {
        self.phase.store(LOADING, Ordering::SeqCst);
    }
}
