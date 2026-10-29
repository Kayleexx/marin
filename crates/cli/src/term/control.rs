use super::app::App;
use crate::CliError;
use crate::assist;

impl App {
    /// `/awareness`, `/context`, `/suggestions` and `/why`: the helper's controls, without leaving the chat. An
    /// accepted suggestion becomes an editable draft in the input line and nothing is sent until Enter.
    pub(super) fn run_control(&mut self, command: &str, words: &str) -> Result<(), CliError> {
        match assist::slash(self.home.as_ref(), command, words) {
            Ok(reply) => {
                self.print_dim(reply.lines)?;
                if let Some((draft, _)) = reply.draft {
                    self.editor.insert(&draft);
                }
            }
            Err(e) => self.show_error(&e)?,
        }
        Ok(())
    }
}
