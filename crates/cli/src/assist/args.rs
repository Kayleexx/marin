use std::path::PathBuf;

use clap::Subcommand;

#[derive(Subcommand, Debug)]
pub enum AssistAction {
    /// Start the background assistant (off unless you start it): rules only, no model, no GPU
    Start {
        /// Also send a desktop notification (notify-send) for a new suggestion
        #[arg(long)]
        notify: bool,
    },
    /// Stop it
    Stop,
    /// Is it running, what it watches and how much it has seen
    Status,
    /// Print a systemd --user unit that runs it (nothing is installed for you)
    Unit,
    /// Allow suggestions for the git repository around PATH [default: the current folder]
    Allow { path: Option<PathBuf> },
    /// Stop watching that repository
    Deny { path: Option<PathBuf> },
    /// Run a command as usual; if it fails in an allowed repo, tell the assistant (`marin assist watch -- cargo test`)
    Watch {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// List the suggestions waiting for you
    Inbox,
    /// Why a suggestion exists, and exactly what accepting it would put in the chat
    Why { id: u32 },
    /// Open a normal chat with the failure as an editable draft (nothing is sent until you press Enter)
    Accept {
        id: u32,
        /// Ask in code mode, where the model can read the project (read-only)
        #[arg(long)]
        code: bool,
    },
    /// Silence a suggestion (the same failure stays quiet for a day)
    Dismiss { id: u32 },
    /// Hide a suggestion for a while
    Snooze {
        id: u32,
        #[arg(long, default_value_t = 60)]
        minutes: u32,
    },
    /// Awareness controls: `status`, `enable`, `disable`, `desktop on|off`, `browser on|off`, `allow-app NAME`,
    /// `allow-site HOST`, `deny-app`, `deny-site`, `notify on|off`, `autostart on|off`
    Awareness {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        words: Vec<String>,
    },
    /// What the helper currently retains (in its memory only)
    Context,
    /// `list` (default), `accept N`, `dismiss N`, `mute N`, `snooze N`, `unmute TOPIC`
    Suggestions {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        words: Vec<String>,
    },
    /// Share a page or window by hand (text may come from stdin); it always gets a suggestion
    Share {
        title: String,
        #[arg(long, default_value = "")]
        url: String,
    },
    /// Firefox tab titles of sites you allowed: `install` or `remove` the link (the extension is in extras/browser)
    Browser {
        what: String,
        #[arg(hide = true, trailing_var_arg = true, allow_hyphen_values = true)]
        from: Vec<String>,
    },
    #[command(hide = true)]
    Daemon {
        #[arg(long)]
        notify: bool,
    },
}
