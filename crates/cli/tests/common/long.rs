//! The tiny chat model with a window long enough for code mode.

use std::fs;
use std::io;
use std::path::Path;

/// A copy of the tiny chat model in `from` at `to` whose window is 8192 tokens.
pub fn long_tiny(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    let config = to.join("config.json");
    let longer = fs::read_to_string(&config)?.replace(
        "\"max_position_embeddings\": 1024",
        "\"max_position_embeddings\": 8192",
    );
    fs::write(config, longer)
}
