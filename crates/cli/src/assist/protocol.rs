use serde::{Deserialize, Serialize};

use super::inbox::{Event, State};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Event(Event),
    Status,
    Stop,
    Mark { id: u32, state: State },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub message: String,
}
