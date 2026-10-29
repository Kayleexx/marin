use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::activity::{Activity, Source, host_and_path};
use super::client;
use super::protocol::Request;
use super::store::Store;
use crate::CliError;

/// Firefox native messaging: a 4-byte length (native byte order) and then JSON, in both directions.
const MAX_FRAME: usize = 16 * 1024;
const HOST_NAME: &str = "marin_awareness";
const EXTENSION_ID: &str = "marin-awareness@marin.local";

#[derive(Deserialize)]
struct Tab {
    url: String,
    #[serde(default)]
    title: String,
}

fn read_frame(input: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut size = [0_u8; 4];
    if input.read_exact(&mut size).is_err() {
        return Ok(None);
    }
    let size = usize::try_from(u32::from_ne_bytes(size)).unwrap_or(usize::MAX);
    if size > MAX_FRAME {
        return Err(io::Error::other("a message from the browser was too large"));
    }
    let mut body = vec![0_u8; size];
    input.read_exact(&mut body)?;
    Ok(Some(body))
}

fn write_frame(out: &mut impl Write, text: &str) -> io::Result<()> {
    let json = serde_json::json!({ "note": text }).to_string();
    let size = u32::try_from(json.len()).map_err(io::Error::other)?;
    out.write_all(&size.to_ne_bytes())?;
    out.write_all(json.as_bytes())?;
    out.flush()
}

/// One tab message: dropped here unless awareness, the browser source and the site are all approved, so a page
/// that is not on the allowlist never reaches the assistant. Returns what to tell the extension.
fn handle(body: &[u8], store: &Store, socket: &Path) -> String {
    let Ok(tab) = serde_json::from_slice::<Tab>(body) else {
        return "not understood".into();
    };
    let Some((host, _)) = host_and_path(&tab.url) else {
        return "ignored: not a web address".into();
    };
    if !store.awareness().site_allowed(&host) {
        return "ignored: site not allowed".into();
    }
    let activity = Activity {
        source: Source::Tab,
        app: String::new(),
        title: tab.title,
        url: tab.url,
        text: String::new(),
        at: 0,
    };
    match client::send(socket, &Request::Activity(activity)) {
        Ok(answer) => answer.message,
        Err(_) => "the assistant is not running".into(),
    }
}

/// The program the browser starts. It lives only as long as the browser keeps it open and uses no CPU between tabs.
pub fn host(store: &Store, socket: &Path, from: &[String]) -> Result<(), CliError> {
    if !from.is_empty() && !from.iter().any(|a| a == EXTENSION_ID) {
        return Err(CliError::BadOption(
            "not started by the Marin extension, so it will not read tab messages".into(),
        ));
    }
    let (mut input, mut out) = (io::stdin().lock(), io::stdout().lock());
    while let Some(body) = read_frame(&mut input)? {
        write_frame(&mut out, &handle(&body, store, socket))?;
    }
    Ok(())
}

fn firefox_dir() -> Result<PathBuf, CliError> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| CliError::BadOption("HOME is not set".into()))?;
    Ok(home.join(".mozilla/native-messaging-hosts"))
}

/// Writes the two small files Firefox needs to start the host: a manifest in its own folder and a launcher in
/// Marin's folder. Nothing else is touched and `remove` deletes both.
pub fn install(store: &Store) -> Result<Vec<String>, CliError> {
    let exe = std::env::current_exe()?;
    let launcher = store.dir().join("browser-host.sh");
    let home = std::env::var_os("MARIN_HOME")
        .map(|h| format!("MARIN_HOME='{}' ", h.to_string_lossy().replace('\'', "")))
        .unwrap_or_default();
    super::store::private_dir(store.dir())?;
    fs::write(
        &launcher,
        format!(
            "#!/bin/sh\nexec {home}'{}' assist browser host \"$@\"\n",
            exe.display()
        ),
    )?;
    fs::set_permissions(&launcher, fs::Permissions::from_mode(0o700))?;
    let manifest = firefox_dir()?.join(format!("{HOST_NAME}.json"));
    fs::create_dir_all(firefox_dir()?)?;
    let json = serde_json::json!({
        "name": HOST_NAME,
        "description": "Marin awareness: tab titles and addresses of sites you allowed",
        "path": launcher,
        "type": "stdio",
        "allowed_extensions": [EXTENSION_ID],
    });
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json).map_err(io::Error::other)?,
    )?;
    Ok(vec![
        format!("wrote {}", manifest.display()),
        format!("wrote {}", launcher.display()),
        "next, the extension: release Firefox only runs signed extensions, so either load extras/browser/ temporarily (about:debugging, This Firefox, Load Temporary Add-on; it is gone after a restart) or sign it yourself (see the README)".into(),
        "then approve each site: /awareness browser on and /awareness allow-site HOST".into(),
    ])
}

pub fn remove(store: &Store) -> Result<Vec<String>, CliError> {
    let manifest = firefox_dir()?.join(format!("{HOST_NAME}.json"));
    let gone = |path: &Path| fs::remove_file(path).is_ok();
    let (a, b) = (gone(&manifest), gone(&store.dir().join("browser-host.sh")));
    Ok(vec![format!(
        "browser link removed ({}); remove the extension in about:addons",
        if a || b {
            "files deleted"
        } else {
            "it was not installed"
        }
    )])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip_and_oversized_ones_are_refused() {
        let mut out = Vec::new();
        write_frame(&mut out, "ok").unwrap();
        let body = read_frame(&mut out.as_slice()).unwrap().unwrap();
        assert_eq!(String::from_utf8_lossy(&body), "{\"note\":\"ok\"}");
        let huge = u32::try_from(MAX_FRAME + 1).unwrap().to_ne_bytes();
        assert!(read_frame(&mut huge.as_slice()).is_err());
        assert!(read_frame(&mut [].as_slice()).unwrap().is_none());
    }
}
