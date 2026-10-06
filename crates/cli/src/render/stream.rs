use super::classify::{Kind, classify};
use super::code::Code;
use super::content::ContentKind;
use super::inline::Inline;
use super::table::{self, Table};
use super::theme::{Look, Theme, looks};
use super::wrap::{Prefix, Wrap};

enum State {
    /// At the start of a source line, waiting to know what it is.
    Start,
    /// Inside a line of running text.
    Text,
    /// Inside a fenced code block.
    Fence,
}

struct Fence {
    code: Code,
    marker: char,
    len: usize,
    at_line_start: bool,
    /// Spaces and fence characters seen at the start of a line: a closing fence if the line ends there.
    probe: String,
}

/// Longest run of leading spaces and fence characters held back while looking for a closing fence.
const PROBE_LIMIT: usize = 40;
const RULE_LIMIT: usize = 48;

/// Markdown turned into terminal text as it arrives, one character at a time.
///
/// Output is append-only: nothing already written is ever revisited, so it can follow a token stream without
/// redrawing. A line's kind is decided from its first few characters (see `classify`); until then they wait.
/// Tables wait for their last row, since column widths need every row.
pub struct MarkdownStream {
    theme: Theme,
    wrap: Wrap,
    inline: Inline,
    state: State,
    line: String,
    lists: Vec<usize>,
    hang: usize,
    blank_pending: bool,
    heading: Option<usize>,
    held: Option<String>,
    table: Option<Table>,
    fence: Option<Fence>,
}

fn spaces(count: usize) -> String {
    " ".repeat(count)
}

impl MarkdownStream {
    /// `margin` columns lead every line; `midline` says the first line is already open at that margin.
    pub fn new(theme: Theme, width: usize, margin: usize, midline: bool) -> Self {
        Self {
            theme,
            wrap: Wrap::new(theme, width, margin, midline),
            inline: Inline::new(),
            state: State::Start,
            line: String::new(),
            lists: Vec::new(),
            hang: 0,
            blank_pending: false,
            heading: None,
            held: None,
            table: None,
            fence: None,
        }
    }

    pub fn take(&mut self) -> String {
        self.wrap.take()
    }

    pub fn feed(&mut self, text: &str) {
        for c in text.chars().filter(|c| *c != '\r') {
            self.push(if c == '\t' { ' ' } else { c });
        }
    }

    fn push(&mut self, c: char) {
        match self.state {
            State::Fence => self.fence_char(c),
            State::Text if c == '\n' => self.end_text(),
            State::Text => self.inline.push(c, &mut self.wrap),
            State::Start if c == '\n' => self.complete_line(),
            State::Start => {
                self.line.push(c);
                self.decide();
            }
        }
    }

    fn leads_with_bar(line: &str) -> bool {
        line.trim_start_matches(' ').starts_with('|')
    }

    fn decide(&mut self) {
        if !Self::leads_with_bar(&self.line) && !self.line.trim().is_empty() {
            self.close_tables();
        }
        if let Some(kind) = classify(&self.line, false)
            && kind != Kind::TableRow
        {
            let line = std::mem::take(&mut self.line);
            self.begin(kind, &line, false);
        }
    }

    fn complete_line(&mut self) {
        let line = std::mem::take(&mut self.line);
        if !Self::leads_with_bar(&line) {
            self.close_tables();
        }
        if let Some(kind) = classify(&line, true) {
            self.begin(kind, &line, true);
        }
    }

    fn begin(&mut self, kind: Kind, line: &str, complete: bool) {
        let blank_prefix = |width: usize| Prefix {
            text: spaces(width),
            width,
        };
        match kind {
            Kind::Blank => self.blank_pending = true,
            Kind::Rule => self.rule(),
            Kind::Fence { marker, len, info } => self.open_fence(marker, len, &info),
            Kind::TableRow => self.table_row(line.to_string()),
            Kind::Paragraph { skip } => {
                let continuation = !self.lists.is_empty() && skip >= 2;
                if !continuation {
                    self.lists.clear();
                    self.hang = 0;
                }
                self.spacing(false);
                let prefix = blank_prefix(self.hang);
                self.text(line, skip, prefix.clone(), prefix, Look::NONE, complete);
            }
            Kind::Heading { level, skip } => {
                self.lists.clear();
                self.hang = 0;
                self.spacing(true);
                self.heading = Some(level);
                let look = match level {
                    1 => looks::H1,
                    2 => looks::H2,
                    _ => looks::H3,
                };
                self.text(
                    line,
                    skip,
                    Prefix::default(),
                    Prefix::default(),
                    look,
                    complete,
                );
            }
            Kind::Quote { skip } => {
                self.lists.clear();
                self.hang = 0;
                self.spacing(false);
                let bar = self
                    .theme
                    .paint(&looks::DIM, if self.theme.unicode { "│" } else { "|" });
                let prefix = Prefix {
                    text: format!("{bar} "),
                    width: 2,
                };
                self.text(line, skip, prefix.clone(), prefix, Look::NONE, complete);
            }
            Kind::Bullet { indent, skip } => {
                let depth = self.enter_list(indent);
                let marker = self.theme.bullet(depth).to_string();
                self.item(&marker, 1, depth, line, skip, complete);
            }
            Kind::Ordered {
                indent,
                number,
                skip,
            } => {
                let depth = self.enter_list(indent);
                let width = number.chars().count();
                self.item(&number, width, depth, line, skip, complete);
            }
        }
    }

    /// Puts a list item at `indent` into the nesting and returns its depth.
    fn enter_list(&mut self, indent: usize) -> usize {
        while self.lists.last().is_some_and(|top| indent < *top) {
            self.lists.pop();
        }
        if self.lists.last() != Some(&indent) {
            self.lists.push(indent);
        }
        self.spacing(false);
        self.lists.len().saturating_sub(1)
    }

    fn item(
        &mut self,
        marker: &str,
        marker_width: usize,
        depth: usize,
        line: &str,
        skip: usize,
        complete: bool,
    ) {
        let lead = depth.saturating_mul(2);
        let width = lead.saturating_add(marker_width).saturating_add(1);
        let first = Prefix {
            text: format!("{}{} ", spaces(lead), self.theme.paint(&looks::H3, marker)),
            width,
        };
        let cont = Prefix {
            text: spaces(width),
            width,
        };
        self.hang = width;
        self.text(line, skip, first, cont, Look::NONE, complete);
    }

    /// Starts a text line and feeds it what has arrived of it.
    fn text(
        &mut self,
        line: &str,
        skip: usize,
        first: Prefix,
        cont: Prefix,
        base: Look,
        complete: bool,
    ) {
        self.wrap.begin(first, cont);
        self.inline.set_base(base);
        for c in line.get(skip..).unwrap_or("").chars() {
            self.inline.push(c, &mut self.wrap);
        }
        if complete {
            self.end_text();
        } else {
            self.state = State::Text;
        }
    }

    fn end_text(&mut self) {
        self.inline.end_line(&mut self.wrap);
        self.wrap.flush_word();
        if let Some(level) = self.heading.take() {
            let columns = self.wrap.line_columns();
            self.wrap.end_line();
            if !self.theme.color && level <= 2 && columns > 0 {
                let rule = if level == 1 { "=" } else { "-" }.repeat(columns);
                self.wrap.line(&rule, columns);
            }
            self.blank_pending = true;
        } else {
            self.wrap.end_line();
        }
        self.state = State::Start;
    }

    /// An empty line before a block that wants one, or one the source asked for.
    fn spacing(&mut self, force: bool) {
        if (self.blank_pending || force) && self.wrap.printed() {
            self.wrap.blank_line();
        }
        self.blank_pending = false;
    }

    fn rule(&mut self) {
        self.lists.clear();
        self.hang = 0;
        self.spacing(true);
        self.wrap.fresh_line();
        let width = self.wrap.room(0).min(RULE_LIMIT);
        let glyph = self.theme.frame().horizontal.repeat(width);
        self.wrap
            .line(&self.theme.paint(&looks::DIM, &glyph), width);
        self.blank_pending = true;
    }

    fn open_fence(&mut self, marker: char, len: usize, info: &str) {
        self.lists.clear();
        self.hang = 0;
        self.spacing(true);
        self.wrap.fresh_line();
        let label = info.split_whitespace().next().unwrap_or("");
        let mut code = Code::new(ContentKind::from_hint(label), label, true, true);
        code.open(&mut self.wrap);
        self.fence = Some(Fence {
            code,
            marker,
            len,
            at_line_start: true,
            probe: String::new(),
        });
        self.state = State::Fence;
    }

    fn fence_char(&mut self, c: char) {
        let Some(fence) = self.fence.as_mut() else {
            self.state = State::Start;
            return;
        };
        if fence.at_line_start {
            if c == '\n' {
                let marks = fence.probe.chars().filter(|p| *p == fence.marker).count();
                let only_marks = fence.probe.chars().all(|p| p == fence.marker || p == ' ');
                if only_marks && marks >= fence.len {
                    self.close_fence(false);
                    return;
                }
            } else if (c == fence.marker || c == ' ') && fence.probe.chars().count() < PROBE_LIMIT {
                fence.probe.push(c);
                return;
            }
            for held in std::mem::take(&mut fence.probe).chars() {
                fence.code.push(held, &mut self.wrap);
            }
            fence.at_line_start = false;
        }
        fence.code.push(c, &mut self.wrap);
        fence.at_line_start = c == '\n';
    }

    /// Ends the block. At the end of the input, a held partial fence line is code after all.
    fn close_fence(&mut self, flush_probe: bool) {
        if let Some(mut fence) = self.fence.take() {
            if flush_probe {
                for held in std::mem::take(&mut fence.probe).chars() {
                    fence.code.push(held, &mut self.wrap);
                }
            }
            fence.code.finish(&mut self.wrap);
        }
        self.state = State::Start;
        self.blank_pending = true;
    }

    fn table_row(&mut self, line: String) {
        if let Some(table) = self.table.as_mut() {
            table.rows.push(table::cells(&line));
            return;
        }
        let Some(header) = self.held.take() else {
            self.held = Some(line);
            return;
        };
        match table::delimiter(&line) {
            Some(aligns) if table::cells(&header).len() == aligns.len() => {
                self.table = Some(Table {
                    header: table::cells(&header),
                    aligns,
                    rows: Vec::new(),
                });
            }
            _ => {
                self.begin(Kind::Paragraph { skip: 0 }, &header, true);
                self.table_row(line);
            }
        }
    }

    /// Writes a finished table, or shows a held `|` line as the text it turned out to be.
    fn close_tables(&mut self) {
        if let Some(table) = self.table.take() {
            self.lists.clear();
            self.spacing(true);
            self.wrap.fresh_line();
            let room = self.wrap.room(0);
            let lines = table::layout(&table, room, &self.theme)
                .unwrap_or_else(|| table::stacked(&table, room));
            for (text, width) in lines {
                if text.is_empty() {
                    self.wrap.blank_line();
                } else {
                    self.wrap.line(&text, width);
                }
            }
            self.blank_pending = true;
        }
        if let Some(held) = self.held.take() {
            self.begin(Kind::Paragraph { skip: 0 }, &held, true);
        }
    }

    /// The input has ended: whatever is half-written is completed, and the output stops at a line start.
    pub fn finish(&mut self) {
        match self.state {
            State::Text => self.end_text(),
            State::Start if !self.line.is_empty() => self.complete_line(),
            State::Start | State::Fence => {}
        }
        if matches!(self.state, State::Text) {
            self.end_text();
        }
        if matches!(self.state, State::Fence) {
            self.close_fence(true);
        }
        self.close_tables();
        self.wrap.finish();
    }
}
