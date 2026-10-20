use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use chat::{ModelKind, classify};
use loader::Weights;
use serde::{Deserialize, Serialize};

use crate::CliError;

const MAX_HISTORY: usize = 500;

/// Where marin keeps the remembered model and prompt history.
#[derive(Debug, Clone)]
pub struct Home {
    config_dir: PathBuf,
    state_dir: PathBuf,
}

/// What is remembered between runs. Flags always win over it, and a missing field is simply not remembered.
#[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<PathBuf>,
    /// `auto`, `cpu` or `cuda`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    /// `int8`, for a packed 8-bit model the user picked: the opt-in belongs to `model` and goes with it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantize: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quant_group: Option<usize>,
}

impl Home {
    pub fn new(config_dir: PathBuf, state_dir: PathBuf) -> Self {
        Self {
            config_dir,
            state_dir,
        }
    }

    /// `MARIN_HOME` wins, then the XDG directories, then `~/.config` and `~/.local/state`.
    pub fn detect() -> Option<Self> {
        let var = |name: &str| {
            env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        if let Some(home) = var("MARIN_HOME") {
            return Some(Self::new(home.clone(), home));
        }
        let user = var("HOME");
        let config = var("XDG_CONFIG_HOME").or_else(|| user.as_ref().map(|h| h.join(".config")))?;
        let state = var("XDG_STATE_HOME").or_else(|| user.map(|h| h.join(".local/state")))?;
        Some(Self::new(config.join("marin"), state.join("marin")))
    }

    fn config_file(&self) -> PathBuf {
        self.config_dir.join("config.json")
    }

    /// Where saved conversations live; nothing is written there unless the user saves one.
    pub fn sessions_dir(&self) -> PathBuf {
        self.state_dir.join("sessions")
    }

    fn history_file(&self) -> PathBuf {
        self.state_dir.join("history")
    }

    /// A missing or unreadable config just means "nothing remembered yet".
    pub fn load_config(&self) -> Config {
        self.read_config().0
    }

    /// The config and, when the file exists but cannot be used, what is wrong with it.
    pub fn read_config(&self) -> (Config, Option<String>) {
        let file = self.config_file();
        let text = match fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return (Config::default(), None),
            Err(e) => return (Config::default(), Some(format!("{}: {e}", file.display()))),
        };
        match serde_json::from_str(&text) {
            Ok(config) => (config, None),
            Err(e) => (
                Config::default(),
                Some(format!("{} is not valid ({e})", file.display())),
            ),
        }
    }

    /// Changes one remembered value and keeps the rest.
    pub fn update_config(&self, change: impl FnOnce(&mut Config)) -> std::io::Result<()> {
        let mut config = self.load_config();
        change(&mut config);
        self.save_config(&config)
    }

    pub fn save_config(&self, config: &Config) -> std::io::Result<()> {
        fs::create_dir_all(&self.config_dir)?;
        let text = serde_json::to_string_pretty(config).map_err(std::io::Error::other)?;
        fs::write(self.config_file(), text + "\n")
    }

    /// One JSON string per line, so multi-line prompts survive.
    pub fn load_history(&self) -> Vec<String> {
        let text = fs::read_to_string(self.history_file()).unwrap_or_default();
        text.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }

    pub fn save_history(&self, entries: &[String]) -> std::io::Result<()> {
        fs::create_dir_all(&self.state_dir)?;
        let start = entries.len().saturating_sub(MAX_HISTORY);
        let mut text = String::new();
        for entry in entries.iter().skip(start) {
            text.push_str(&serde_json::to_string(entry).map_err(std::io::Error::other)?);
            text.push('\n');
        }
        fs::write(self.history_file(), text)
    }
}

pub fn is_model_dir(path: &Path) -> bool {
    path.join("config.json").is_file()
        && path.join("tokenizer.json").is_file()
        && Weights::exists_in(path)
}

/// `MARIN_MODELS` (a path list) or `~/models`.
pub fn model_roots() -> Vec<PathBuf> {
    match env::var_os("MARIN_MODELS").filter(|v| !v.is_empty()) {
        Some(list) => env::split_paths(&list).collect(),
        None => env::var_os("HOME")
            .map(|h| vec![PathBuf::from(h).join("models")])
            .unwrap_or_default(),
    }
}

/// Model folders directly inside each root, sorted by name.
pub fn discover(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = roots
        .iter()
        .filter_map(|root| fs::read_dir(root).ok())
        .flatten()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| is_model_dir(path))
        .collect();
    found.sort();
    found
}

pub fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

pub fn model_kind(path: &Path) -> ModelKind {
    classify(
        fs::read_to_string(path.join("tokenizer_config.json"))
            .ok()
            .as_deref(),
    )
}

#[derive(Debug, PartialEq, Eq)]
pub enum Choice {
    Use(PathBuf),
    /// Chat models found on disk; the caller picks one.
    Pick(Vec<PathBuf>),
}

/// Picks the model `marin` talks to. Only chat models qualify: a base model would just continue the text.
/// An explicit path that is not a model folder passes through, so loading reports exactly what is wrong.
pub fn choose_chat_model(
    explicit: Option<&Path>,
    remembered: Option<&Path>,
    roots: &[PathBuf],
) -> Result<Choice, CliError> {
    if let Some(path) = explicit {
        return match is_model_dir(path) && model_kind(path) != ModelKind::Chat {
            true => Err(CliError::NotAChatModel(display_name(path))),
            false => Ok(Choice::Use(path.to_path_buf())),
        };
    }
    if let Some(path) = remembered.filter(|p| is_model_dir(p) && model_kind(p) == ModelKind::Chat) {
        return Ok(Choice::Use(path.to_path_buf()));
    }
    let (chat, base): (Vec<PathBuf>, Vec<PathBuf>) = discover(roots)
        .into_iter()
        .partition(|path| model_kind(path) == ModelKind::Chat);
    if chat.is_empty() {
        return Err(CliError::NoChatModel {
            base: base.iter().map(|p| display_name(p)).collect(),
        });
    }
    Ok(Choice::Pick(chat))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Temp(PathBuf);

    impl Temp {
        fn new(name: &str) -> Self {
            let path = env::temp_dir().join(format!("marin-config-{}-{name}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fake_model(root: &Path, name: &str, chat: bool) -> PathBuf {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        for file in ["config.json", "tokenizer.json", "model.safetensors"] {
            fs::write(dir.join(file), b"x").unwrap();
        }
        let eos = if chat { "<|im_end|>" } else { "<|endoftext|>" };
        let config = format!(r#"{{"eos_token": "{eos}", "chat_template": "<|im_start|>"}}"#);
        fs::write(dir.join("tokenizer_config.json"), config).unwrap();
        dir
    }

    #[test]
    fn config_and_history_round_trip() {
        let temp = Temp::new("roundtrip");
        let home = Home::new(temp.0.join("c"), temp.0.join("s"));
        assert_eq!(home.load_config(), Config::default());
        let config = Config {
            model: Some(PathBuf::from("/models/x")),
            backend: Some("cpu".into()),
            temp: Some(0.2),
            max_tokens: Some(256),
            quantize: Some("int8".into()),
            quant_group: Some(128),
        };
        home.save_config(&config).unwrap();
        assert_eq!(home.load_config(), config);
        let entries = vec!["one".to_string(), "two\nlines".to_string()];
        home.save_history(&entries).unwrap();
        assert_eq!(home.load_history(), entries);
    }

    #[test]
    fn an_old_config_with_only_a_model_still_loads_and_updates_keep_the_rest() {
        let temp = Temp::new("upgrade");
        let home = Home::new(temp.0.clone(), temp.0.clone());
        fs::write(
            temp.0.join("config.json"),
            "{\n  \"model\": \"/models/old\"\n}\n",
        )
        .unwrap();
        let (config, damaged) = home.read_config();
        assert!(damaged.is_none());
        assert_eq!(config.model, Some(PathBuf::from("/models/old")));
        assert_eq!((config.backend, config.temp), (None, None));
        home.update_config(|c| c.backend = Some("cuda".into()))
            .unwrap();
        let after = home.load_config();
        assert_eq!(after.model, Some(PathBuf::from("/models/old")));
        assert_eq!(after.backend.as_deref(), Some("cuda"));
        let text = fs::read_to_string(temp.0.join("config.json")).unwrap();
        assert!(
            !text.contains("temp"),
            "unset values are not written: {text}"
        );
    }

    #[test]
    fn a_damaged_config_is_reported_and_replaced_on_the_next_save() {
        let temp = Temp::new("damaged");
        let home = Home::new(temp.0.clone(), temp.0.clone());
        fs::write(temp.0.join("config.json"), "{not json").unwrap();
        let (config, damaged) = home.read_config();
        assert_eq!(config, Config::default());
        assert!(damaged.is_some_and(|m| m.contains("not valid")));
        home.update_config(|c| c.model = Some(PathBuf::from("/m")))
            .unwrap();
        assert!(home.read_config().1.is_none());
    }

    #[test]
    fn corrupt_config_and_history_are_ignored() {
        let temp = Temp::new("corrupt");
        let home = Home::new(temp.0.clone(), temp.0.clone());
        fs::write(temp.0.join("config.json"), "{not json").unwrap();
        fs::write(temp.0.join("history"), "\"ok\"\ngarbage\n\"also ok\"\n").unwrap();
        assert_eq!(home.load_config(), Config::default());
        assert_eq!(home.load_history(), ["ok", "also ok"]);
    }

    #[test]
    fn history_is_capped_to_the_newest_entries() {
        let temp = Temp::new("cap");
        let home = Home::new(temp.0.clone(), temp.0.clone());
        let entries: Vec<String> = (0..MAX_HISTORY + 20).map(|i| i.to_string()).collect();
        home.save_history(&entries).unwrap();
        let loaded = home.load_history();
        assert_eq!(loaded.len(), MAX_HISTORY);
        assert_eq!(loaded.first().map(String::as_str), Some("20"));
    }

    #[test]
    fn discovery_finds_complete_model_folders_only() {
        let temp = Temp::new("discover");
        let root = temp.0.join("models");
        fake_model(&root, "b-model", true);
        fake_model(&root, "a-model", true);
        fs::create_dir_all(root.join("incomplete")).unwrap();
        fs::write(root.join("incomplete/config.json"), b"x").unwrap();
        let names: Vec<String> = discover(&[root, temp.0.join("missing")])
            .iter()
            .map(|p| display_name(p))
            .collect();
        assert_eq!(names, ["a-model", "b-model"]);
    }

    #[test]
    fn only_chat_models_are_chosen_and_base_models_are_explained() {
        let temp = Temp::new("choose");
        let root = temp.0.join("models");
        let chat = fake_model(&root, "chat", true);
        let base = fake_model(&root, "base", false);
        let roots = std::slice::from_ref(&root);

        assert_eq!(
            choose_chat_model(None, None, roots).unwrap(),
            Choice::Pick(vec![chat.clone()])
        );
        assert_eq!(
            choose_chat_model(None, Some(&chat), &[]).unwrap(),
            Choice::Use(chat.clone())
        );
        assert_eq!(
            choose_chat_model(Some(&chat), None, &[]).unwrap(),
            Choice::Use(chat.clone())
        );
        assert!(matches!(
            choose_chat_model(Some(&base), None, roots),
            Err(CliError::NotAChatModel(name)) if name == "base"
        ));
        let remembered_base = choose_chat_model(None, Some(&base), roots).unwrap();
        assert_eq!(
            remembered_base,
            Choice::Pick(vec![chat]),
            "a remembered base model is ignored"
        );
        let explicit_missing = temp.0.join("not-there");
        assert_eq!(
            choose_chat_model(Some(&explicit_missing), None, &[]).unwrap(),
            Choice::Use(explicit_missing)
        );
    }

    #[test]
    fn with_only_base_models_the_error_names_them_and_nothing_is_picked() {
        let temp = Temp::new("onlybase");
        let root = temp.0.join("models");
        fake_model(&root, "Qwen3-1.7B-Base", false);
        let error = choose_chat_model(None, None, &[root]).unwrap_err();
        let text = error.to_string();
        assert!(
            text.contains("no chat model found")
                && text.contains("Qwen3-1.7B-Base is a base model"),
            "{text}"
        );
        assert!(error.hint().is_some_and(|h| h.contains("Qwen/Qwen3-1.7B")));
        let nothing = choose_chat_model(None, None, &[temp.0.join("empty")]).unwrap_err();
        assert_eq!(nothing.to_string(), "no chat model found");
    }
}
