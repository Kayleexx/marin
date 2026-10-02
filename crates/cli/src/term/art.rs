use super::caps::{Caps, Color};

const MAGIC: &[u8; 4] = b"MPG1";
const HEADER: usize = 6;
const CELL_BYTES: usize = 9;
const RAMP: &[u8] = b" .:-=+*#%@";
/// Cells darker than this use the terminal's own background, so art on black blends into any dark theme.
const TRANSPARENT_BELOW: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cell {
    glyph: char,
    /// How much of the cell the glyph covers, 0 (blank) to 255 (solid).
    ink: u8,
    fg: [u8; 3],
    bg: [u8; 3],
}

/// Startup art made by `scripts/make_portrait.py`: a grid of terminal glyphs (quadrant blocks, braille, halves),
/// each with its own foreground and background color, matched to the source image at development time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Art {
    cols: usize,
    rows: usize,
    cells: Vec<Cell>,
    blend_dark_background: bool,
}

/// Only blank, block-element and braille glyphs are accepted, so an asset can never carry control bytes.
fn allowed(glyph: char) -> bool {
    glyph == ' '
        || ('\u{2580}'..='\u{259f}').contains(&glyph)
        || ('\u{2800}'..='\u{28ff}').contains(&glyph)
}

impl Art {
    /// The full composition at its largest, 96x44 cells, for big terminals.
    pub fn portrait_xl() -> Option<Self> {
        Self::parse(include_bytes!("../../assets/portrait-xl.bin"), true)
    }

    /// The full composition, 68x31 cells.
    pub fn portrait_large() -> Option<Self> {
        Self::parse(include_bytes!("../../assets/portrait-large.bin"), true)
    }

    /// A face-focused crop, 46x21 cells, for terminals too small for the full composition.
    pub fn portrait_medium() -> Option<Self> {
        Self::parse(include_bytes!("../../assets/portrait-medium.bin"), true)
    }

    /// The textured wordmark, 44x8 cells.
    pub fn wordmark_wide() -> Option<Self> {
        Self::parse(include_bytes!("../../assets/wordmark-wide.bin"), true)
    }

    /// The same wordmark at half size, 22x4 cells, for narrow terminals.
    pub fn wordmark_narrow() -> Option<Self> {
        Self::parse(include_bytes!("../../assets/wordmark-narrow.bin"), true)
    }

    pub fn parse(bytes: &[u8], blend_dark_background: bool) -> Option<Self> {
        let (header, body) = bytes.split_at_checked(HEADER)?;
        let (magic, size) = header.split_at_checked(MAGIC.len())?;
        let [cols, rows] = <[u8; 2]>::try_from(size).ok()?;
        let (cols, rows) = (usize::from(cols), usize::from(rows));
        let count = cols.checked_mul(rows)?;
        if magic != MAGIC || count == 0 || body.len() != count.checked_mul(CELL_BYTES)? {
            return None;
        }
        let cells = body
            .chunks_exact(CELL_BYTES)
            .map(|c| match c {
                [lo, hi, ink, fr, fg, fb, br, bg, bb] => {
                    let glyph = char::from_u32(u32::from(u16::from_le_bytes([*lo, *hi])))?;
                    allowed(glyph).then_some(Cell {
                        glyph,
                        ink: *ink,
                        fg: [*fr, *fg, *fb],
                        bg: [*br, *bg, *bb],
                    })
                }
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            cols,
            rows,
            cells,
            blend_dark_background,
        })
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Each returned row is exactly `cols` terminal columns wide.
    pub fn render(&self, caps: &Caps) -> Vec<String> {
        let colored = caps.color != Color::None && caps.unicode;
        let range = self.brightness_range();
        self.cells
            .chunks_exact(self.cols)
            .map(|row| {
                if colored {
                    self.colored_row(row, caps)
                } else {
                    row.iter()
                        .map(|cell| ramp_char(brightness(cell), range))
                        .collect()
                }
            })
            .collect()
    }

    fn colored_row(&self, row: &[Cell], caps: &Caps) -> String {
        let mut out = String::new();
        let mut fg: Option<[u8; 3]> = None;
        let mut bg: Option<Option<[u8; 3]>> = None;
        for cell in row {
            let cell_bg = (!(self.blend_dark_background && luma(cell.bg) < TRANSPARENT_BELOW))
                .then_some(cell.bg);
            // A blank cell shows only its background and a solid one only its foreground.
            if cell.glyph != ' ' && fg != Some(cell.fg) {
                out.push_str(&caps.fg(cell.fg));
                fg = Some(cell.fg);
            }
            if cell.ink != u8::MAX && bg != Some(cell_bg) {
                match cell_bg {
                    Some(color) => out.push_str(&caps.bg(color)),
                    None => out.push_str(caps.default_bg()),
                }
                bg = Some(cell_bg);
            }
            out.push(cell.glyph);
        }
        out.push_str(caps.reset());
        out
    }

    /// The 5th and 98th percentile of cell brightness, so the ASCII ramp uses its whole range
    /// even though the art is mostly dark.
    fn brightness_range(&self) -> (f32, f32) {
        let mut levels: Vec<f32> = self.cells.iter().map(brightness).collect();
        levels.sort_by(f32::total_cmp);
        let at = |fraction: f32| {
            levels
                .get(((levels.len() as f32) * fraction) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        (at(0.05), at(0.98))
    }
}

fn luma(color: [u8; 3]) -> f32 {
    (0.2126 * f32::from(color[0]) + 0.7152 * f32::from(color[1]) + 0.0722 * f32::from(color[2]))
        / 255.0
}

fn brightness(cell: &Cell) -> f32 {
    let ink = f32::from(cell.ink) / 255.0;
    luma(cell.bg) * (1.0 - ink) + luma(cell.fg) * ink
}

fn ramp_char(level: f32, (low, high): (f32, f32)) -> char {
    let last = RAMP.len().saturating_sub(1);
    let spread = (high - low).max(f32::EPSILON);
    let index = (((level - low) / spread).clamp(0.0, 1.0) * last as f32).round() as usize;
    RAMP.get(index.min(last)).map_or(' ', |b| char::from(*b))
}

#[cfg(test)]
pub(super) fn synthetic(cols: u8, rows: u8) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend([cols, rows]);
    for row in 0..rows {
        for col in 0..cols {
            let shade = col.wrapping_mul(7).wrapping_add(row.wrapping_mul(5));
            let glyph = 0x2800_u16.saturating_add(u16::from(shade));
            bytes.extend(glyph.to_le_bytes());
            bytes.extend([shade / 2, 40, shade, 250, 10, 20, shade / 2]);
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::super::caps::strip_ansi;
    use super::super::screen::display_width;
    use super::*;

    fn caps(color: Color, unicode: bool) -> Caps {
        Caps {
            color,
            unicode,
            dumb: false,
        }
    }

    fn all() -> [(&'static str, Art, usize, usize); 5] {
        [
            ("xl", Art::portrait_xl().unwrap(), 96, 44),
            ("large", Art::portrait_large().unwrap(), 68, 31),
            ("medium", Art::portrait_medium().unwrap(), 46, 21),
            ("wide wordmark", Art::wordmark_wide().unwrap(), 44, 8),
            ("narrow wordmark", Art::wordmark_narrow().unwrap(), 22, 4),
        ]
    }

    #[test]
    fn well_formed_data_parses_and_malformed_data_is_rejected() {
        let good = synthetic(6, 3);
        let art = Art::parse(&good, false).unwrap();
        assert_eq!((art.cols(), art.rows()), (6, 3));
        assert!(Art::parse(&[], false).is_none());
        assert!(
            Art::parse(&good[..good.len() - 1], false).is_none(),
            "truncated"
        );
        let mut bad_magic = good.clone();
        bad_magic[0] = b'X';
        assert!(Art::parse(&bad_magic, false).is_none());
        let mut zero = good;
        zero[4] = 0;
        assert!(Art::parse(&zero, false).is_none());
    }

    #[test]
    fn glyphs_outside_blocks_and_braille_are_rejected_so_assets_cannot_carry_control_bytes() {
        for glyph in [0x1b_u16, 0x41, 0x0a, 0x2500, 0x2600, 0x2900] {
            let mut bytes = synthetic(2, 1);
            bytes.splice(HEADER..HEADER + 2, glyph.to_le_bytes());
            assert!(Art::parse(&bytes, false).is_none(), "{glyph:#x}");
        }
        let mut blank = synthetic(2, 1);
        blank.splice(HEADER..HEADER + 2, 0x20_u16.to_le_bytes());
        assert!(Art::parse(&blank, false).is_some());
    }

    #[test]
    fn the_bundled_assets_have_the_documented_sizes_and_real_tonal_range() {
        for (name, art, cols, rows) in all() {
            assert_eq!((art.cols(), art.rows()), (cols, rows), "{name}");
            let colors: std::collections::BTreeSet<[u8; 3]> =
                art.cells.iter().flat_map(|c| [c.fg, c.bg]).collect();
            assert!(
                colors.len() > 100,
                "{name}: only {} distinct colors",
                colors.len()
            );
            let (low, high) = art.brightness_range();
            assert!(high - low > 0.1, "{name}: no tonal range {low}..{high}");
            let glyphs: std::collections::BTreeSet<char> =
                art.cells.iter().map(|c| c.glyph).collect();
            assert!(
                glyphs.len() >= 12,
                "{name}: glyph texture needs variety, got {}",
                glyphs.len()
            );
        }
    }

    #[test]
    fn every_rendered_row_is_exactly_cols_columns_wide_in_every_mode() {
        let modes = [
            caps(Color::Truecolor, true),
            caps(Color::Ansi256, true),
            caps(Color::None, true),
            caps(Color::Truecolor, false),
        ];
        for (name, art, cols, rows) in all() {
            for mode in modes {
                let lines = art.render(&mode);
                assert_eq!(lines.len(), rows, "{name}");
                for line in lines {
                    assert_eq!(display_width(&strip_ansi(&line)), cols, "{name} {mode:?}");
                }
            }
        }
    }

    #[test]
    fn monochrome_and_non_unicode_output_use_plain_ascii_and_no_escapes() {
        for (name, art, _, _) in all() {
            for mode in [caps(Color::None, true), caps(Color::Truecolor, false)] {
                for row in art.render(&mode) {
                    assert!(row.is_ascii() && !row.contains('\x1b'), "{name}: {row:?}");
                }
            }
        }
    }

    #[test]
    fn truecolor_rows_use_braille_and_block_glyphs_with_sequences_only_when_colors_change() {
        let art = Art::portrait_large().unwrap();
        let rows = art.render(&caps(Color::Truecolor, true));
        let text = rows.concat();
        assert!(
            text.chars().any(|c| ('\u{2800}'..='\u{28ff}').contains(&c)),
            "braille texture"
        );
        assert!(
            art.cells
                .iter()
                .all(|c| c.glyph == ' ' || ('\u{2800}'..='\u{28ff}').contains(&c.glyph)),
            "portraits are thin braille glyphs only, no block rectangles"
        );
        let glyph_count = art.cells.iter().filter(|c| c.glyph != ' ').count();
        assert!(
            text.matches("\x1b[38;2").count() <= glyph_count,
            "no redundant foreground codes"
        );
        assert!(rows.iter().all(|r| r.ends_with("\x1b[0m")));
    }

    #[test]
    fn every_asset_lets_the_terminal_background_show_through() {
        let truecolor = caps(Color::Truecolor, true);
        let wordmark = Art::wordmark_wide().unwrap().render(&truecolor).concat();
        assert!(
            wordmark.contains("\x1b[49m"),
            "dark cells use the default background"
        );
        let portrait = Art::portrait_large().unwrap().render(&truecolor).concat();
        assert!(
            portrait.contains("\x1b[49m") && !portrait.contains("\x1b[48"),
            "portraits are foreground glyphs on the terminal's own background"
        );
    }

    #[test]
    fn the_portraits_stay_in_the_cool_palette_with_no_pink() {
        for art in [
            Art::portrait_large().unwrap(),
            Art::portrait_medium().unwrap(),
        ] {
            let pink = art
                .cells
                .iter()
                .flat_map(|c| [c.fg, c.bg])
                .filter(|[r, g, b]| {
                    *r > 150 && *r > g.saturating_add(60) && *b > g.saturating_add(30)
                })
                .count();
            assert_eq!(pink, 0, "no magenta/pink cells");
        }
    }

    #[test]
    fn the_ascii_fallback_uses_the_whole_ramp_even_for_a_dark_portrait() {
        let used: std::collections::BTreeSet<char> = Art::portrait_large()
            .unwrap()
            .render(&caps(Color::None, true))
            .concat()
            .chars()
            .collect();
        assert!(
            used.len() >= 8,
            "only {} distinct ramp characters: {used:?}",
            used.len()
        );
    }

    #[test]
    fn ramp_maps_dark_to_sparse_and_light_to_dense_across_the_measured_range() {
        let range = (0.0, 1.0);
        assert_eq!((ramp_char(0.0, range), ramp_char(1.0, range)), (' ', '@'));
        assert_eq!(ramp_char(0.3, (0.2, 0.4)), '+');
    }
}
