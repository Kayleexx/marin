use serde::{Deserialize, Serialize};

/// What the user approved. Nothing is observed until `enabled` is set and the source of an event is approved.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Awareness {
    pub enabled: bool,
    /// Also send a desktop notification (notify-send) for a new suggestion.
    pub notify: bool,
    pub desktop: bool,
    pub browser: bool,
    /// Tab events count only for these hosts (and their subdomains).
    pub sites: Vec<String>,
    /// Window events count only for these application classes (lowercase).
    pub apps: Vec<String>,
    /// Repos or hosts the user muted; nothing is suggested for them.
    pub muted: Vec<String>,
}

pub const MAX_LIST: usize = 64;

impl Awareness {
    pub fn site_allowed(&self, host: &str) -> bool {
        self.enabled
            && self.browser
            && self
                .sites
                .iter()
                .any(|s| host == s || host.ends_with(&format!(".{s}")))
    }

    pub fn app_allowed(&self, app: &str) -> bool {
        self.enabled && self.desktop && self.apps.iter().any(|a| a.eq_ignore_ascii_case(app))
    }

    pub fn is_muted(&self, topic: &str) -> bool {
        self.muted.iter().any(|m| m == topic)
    }

    pub fn summary(&self) -> Vec<String> {
        let on = |b: bool| if b { "on" } else { "off" };
        vec![
            format!("awareness: {}", on(self.enabled)),
            format!(
                "  desktop windows (experimental, X11 only, unverified on a real X11 session): {} · apps: {}",
                on(self.desktop),
                list(&self.apps)
            ),
            format!(
                "  browser tabs (title and address, never page text): {} · sites: {}",
                on(self.browser),
                list(&self.sites)
            ),
            format!("  notifications (notify-send): {}", on(self.notify)),
            format!("  muted: {}", list(&self.muted)),
        ]
    }

    /// Applies one `desktop|browser on|off` or `allow-app|allow-site|deny-app|deny-site|mute|unmute VALUE` change.
    pub fn change(&mut self, what: &str, value: &str) -> Result<(), String> {
        let flag = match (what, value) {
            ("desktop", "on" | "off") => Some(&mut self.desktop),
            ("browser", "on" | "off") => Some(&mut self.browser),
            ("notify", "on" | "off") => Some(&mut self.notify),
            _ => None,
        };
        if let Some(flag) = flag {
            *flag = value == "on";
            return Ok(());
        }
        let value = value.trim().to_ascii_lowercase();
        if value.is_empty() || value.len() > 200 || value.contains(char::is_whitespace) {
            return Err("give one word: a host, an application class or a repo path".into());
        }
        let (list, add) = match what {
            "allow-site" => (&mut self.sites, true),
            "deny-site" => (&mut self.sites, false),
            "allow-app" => (&mut self.apps, true),
            "deny-app" => (&mut self.apps, false),
            "mute" => (&mut self.muted, true),
            "unmute" => (&mut self.muted, false),
            _ => {
                return Err(format!(
                    "unknown `{what}`: try status, enable, disable, desktop on|off, browser on|off, allow-site HOST, allow-app NAME, deny-site, deny-app, notify on|off, autostart on|off"
                ));
            }
        };
        if add && !list.contains(&value) {
            if list.len() >= MAX_LIST {
                return Err(format!("at most {MAX_LIST} entries"));
            }
            list.push(value);
        } else if !add {
            list.retain(|v| *v != value);
        }
        Ok(())
    }
}

fn list(items: &[String]) -> String {
    if items.is_empty() {
        "none".into()
    } else {
        items.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_approved_sources_hosts_and_apps_count() {
        let mut a = Awareness::default();
        assert!(!a.site_allowed("docs.rs"));
        a.change("allow-site", "docs.rs").unwrap();
        a.change("browser", "on").unwrap();
        assert!(!a.site_allowed("docs.rs"), "awareness itself is still off");
        a.enabled = true;
        assert!(a.site_allowed("docs.rs") && a.site_allowed("x.docs.rs"));
        assert!(!a.site_allowed("evildocs.rs"));
        a.change("allow-app", "Code").unwrap();
        assert!(!a.app_allowed("code"));
        a.change("desktop", "on").unwrap();
        assert!(a.app_allowed("CODE"));
        a.change("deny-site", "docs.rs").unwrap();
        assert!(!a.site_allowed("docs.rs"));
        assert!(a.change("allow-site", "two words").is_err());
        assert!(a.change("nope", "x").is_err());
    }
}
