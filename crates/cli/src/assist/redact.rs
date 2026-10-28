const SECRET_WORDS: [&str; 7] = [
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "auth",
    "credential",
];
const SECRET_PREFIXES: [&str; 8] = [
    "ghp_",
    "gho_",
    "ghs_",
    "github_pat_",
    "glpat-",
    "sk-",
    "xox",
    "AKIA",
];
const MASK: &str = "[redacted]";

/// Removes what looks like a credential: private key blocks, `name=value` pairs whose name suggests a secret,
/// values after `Authorization:` or `Bearer`, well-known token prefixes and long opaque strings. Text that is
/// merely long or odd is kept; a few harmless strings (such as full commit ids) are lost.
pub fn redact(text: &str) -> String {
    let mut in_key = false;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.contains("-----BEGIN") && line.contains("PRIVATE KEY") {
            in_key = true;
        }
        if in_key {
            in_key = !line.contains("-----END");
            out.push(MASK.to_string());
        } else {
            out.push(redact_line(line));
        }
    }
    out.join("\n")
}

fn redact_line(line: &str) -> String {
    let mut pending = 0_u8;
    let mut words = Vec::new();
    for word in line.split(' ') {
        let bare = word.trim_matches(|c: char| "\"'`,;()[]".contains(c));
        if pending > 0 && !bare.is_empty() {
            if !matches!(
                bare.to_ascii_lowercase().as_str(),
                "bearer" | "basic" | "token"
            ) {
                pending = pending.saturating_sub(1);
            }
            words.push(MASK.to_string());
        } else if bare.eq_ignore_ascii_case("bearer") {
            pending = 1;
            words.push(word.to_string());
        } else if let Some((name, value)) = bare.split_once(['=', ':'])
            && is_secret_name(name)
            && !value.starts_with("//")
        {
            let sep = bare
                .get(name.len()..name.len().saturating_add(1))
                .unwrap_or("=");
            pending = u8::from(value.is_empty());
            let shown = if value.is_empty() { "" } else { MASK };
            words.push(format!("{name}{sep}{shown}"));
        } else if looks_secret(bare) {
            words.push(MASK.to_string());
        } else {
            words.push(word.to_string());
        }
    }
    words.join(" ")
}

fn is_secret_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SECRET_WORDS.iter().any(|w| name.contains(w))
}

fn looks_secret(word: &str) -> bool {
    if word.len() >= 12 && SECRET_PREFIXES.iter().any(|p| word.starts_with(p)) {
        return true;
    }
    word.len() >= 32
        && word
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "+/=_-".contains(c))
        && word.chars().any(|c| c.is_ascii_digit())
        && word.chars().any(|c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_secrets_and_known_prefixes_are_masked() {
        let text = "API_KEY=abc123 password: hunter2 ok ghp_abcdefghijklmnop done";
        let out = redact(text);
        assert!(!out.contains("abc123") && !out.contains("hunter2") && !out.contains("ghp_"));
        assert!(out.contains("ok") && out.contains("done"));
    }

    #[test]
    fn authorization_headers_lose_their_value() {
        let out = redact("Authorization: Bearer eyJhbGciOi.payload.sig next");
        assert!(!out.contains("eyJ") && !out.contains("payload"));
        assert!(out.ends_with("next"));
        assert!(!redact("curl -H 'Authorization: Basic dXNlcjpwYXNz'").contains("dXNlcjpwYXNz"));
    }

    #[test]
    fn private_key_blocks_are_removed_whole() {
        let out = redact(
            "before\n-----BEGIN RSA PRIVATE KEY-----\nMIIE\nabc\n-----END RSA PRIVATE KEY-----\nafter",
        );
        assert_eq!(
            out,
            format!("before\n{MASK}\n{MASK}\n{MASK}\n{MASK}\nafter")
        );
    }

    #[test]
    fn long_opaque_strings_go_but_ordinary_text_stays() {
        let out = redact(
            "id 3f786850e387550fdab836ed7e6dc881de23001b stays? error[E0382]: borrow of moved value",
        );
        assert!(!out.contains("3f786850"));
        assert!(out.contains("error[E0382]: borrow of moved value"));
        assert_eq!(
            redact("https://example.com/a key"),
            "https://example.com/a key"
        );
    }
}
