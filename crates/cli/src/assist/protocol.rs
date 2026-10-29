use serde::{Deserialize, Serialize};

use super::activity::Activity;
use super::inbox::{Event, State};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    Event(Event),
    Activity(Activity),
    Context,
    Status,
    Stop,
    Mark { id: u32, state: State },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    pub message: String,
}
