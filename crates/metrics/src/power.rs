use std::fs;

/// Whether the machine runs on mains power, from what `/sys/class/power_supply` says: `Some(true)` when any mains
/// or USB-C supply is online, `Some(false)` when there is a battery and no supply is online, `None` when there is
/// nothing to read (not Linux, or a desktop that reports no supplies).
fn ac_from(supplies: &[(String, String)]) -> Option<bool> {
    let online_supply = |(kind, online): &(String, String)| {
        online.trim() == "1"
            && matches!(
                kind.trim(),
                "Mains" | "USB" | "USB_C" | "USB_PD" | "USB_PD_DRP"
            )
    };
    if supplies.iter().any(online_supply) {
        return Some(true);
    }
    supplies
        .iter()
        .any(|(kind, _)| kind.trim() == "Battery")
        .then_some(false)
}

pub fn on_ac_power() -> Option<bool> {
    let read =
        |dir: &std::path::Path, file: &str| fs::read_to_string(dir.join(file)).unwrap_or_default();
    let supplies: Vec<(String, String)> = fs::read_dir("/sys/class/power_supply")
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| {
            let dir = entry.path();
            (read(&dir, "type"), read(&dir, "online"))
        })
        .collect();
    ac_from(&supplies)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn supply(kind: &str, online: &str) -> (String, String) {
        (kind.into(), online.into())
    }

    #[test]
    fn a_mains_supply_that_is_online_means_ac() {
        let laptop = [
            supply("Mains\n", "1\n"),
            supply("Battery\n", ""),
            supply("USB\n", "0\n"),
        ];
        assert_eq!(ac_from(&laptop), Some(true));
        assert_eq!(
            ac_from(&[supply("USB_PD", "1"), supply("Battery", "")]),
            Some(true),
            "USB-C power counts"
        );
    }

    #[test]
    fn a_battery_with_no_supply_online_means_battery_power() {
        assert_eq!(
            ac_from(&[supply("Mains", "0"), supply("Battery", "")]),
            Some(false)
        );
        assert_eq!(ac_from(&[supply("Battery", "")]), Some(false));
    }

    #[test]
    fn nothing_to_read_is_unknown_not_a_guess() {
        assert_eq!(ac_from(&[]), None);
        assert_eq!(
            ac_from(&[supply("Mains", "0")]),
            None,
            "a supply that is off, with no battery, says nothing"
        );
        assert_eq!(ac_from(&[supply("Wireless", "1")]), None);
    }
}
