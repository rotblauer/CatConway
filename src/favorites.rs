use std::fs;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use crate::grid::Rules;
use crate::search::{parse_rule_label, rules_to_label};

/// Default file path for persisting favorites.
const DEFAULT_FAVORITES_PATH: &str = "favorites.txt";

const FAVORITES_HEADER: &str = "\
# CatConway Favorite Rules
# Format: B<birth>/S<survival>[/R<radius>]
# These rules have been manually marked as interesting.
";

/// A single favorited rule.
#[derive(Debug, Clone)]
pub struct FavoriteRule {
    pub rules: Rules,
    pub label: String,
}

/// Manages the collection of favorited rules with file persistence.
#[derive(Debug)]
pub struct Favorites {
    entries: Vec<FavoriteRule>,
    path: PathBuf,
}

impl Favorites {
    /// Create a new favorites store, loading any existing entries from disk.
    pub fn new() -> Self {
        let path = PathBuf::from(DEFAULT_FAVORITES_PATH);
        let entries = load_favorites(&path);
        Self { entries, path }
    }

    /// Create a favorites store with a custom file path (useful for tests).
    pub fn with_path(path: PathBuf) -> Self {
        let entries = load_favorites(&path);
        Self { entries, path }
    }

    /// Return all favorited rules.
    pub fn entries(&self) -> &[FavoriteRule] {
        &self.entries
    }

    /// Check whether a rule set is currently favorited.
    pub fn is_favorite(&self, rules: &Rules) -> bool {
        self.entries.iter().any(|f| f.rules == *rules)
    }

    /// Add a rule to favorites. Returns `true` if it was added (not a duplicate).
    pub fn add(&mut self, rules: Rules) -> bool {
        if self.is_favorite(&rules) {
            return false;
        }
        let label = rules_to_label(&rules);
        self.entries.push(FavoriteRule {
            rules,
            label: label.clone(),
        });
        self.save();
        true
    }

    /// Remove a rule from favorites. Returns `true` if it was found and removed.
    pub fn remove(&mut self, rules: &Rules) -> bool {
        let before = self.entries.len();
        self.entries.retain(|f| f.rules != *rules);
        if self.entries.len() < before {
            self.save();
            true
        } else {
            false
        }
    }

    /// Toggle a rule's favorite status. Returns `true` if the rule is now a favorite.
    pub fn toggle(&mut self, rules: &Rules) -> bool {
        if self.is_favorite(rules) {
            self.remove(rules);
            false
        } else {
            self.add(*rules);
            true
        }
    }

    /// Persist the current favorites list to disk.
    fn save(&self) {
        save_favorites(&self.path, &self.entries);
    }
}

/// Load favorites from a file.
fn load_favorites(path: &Path) -> Vec<FavoriteRule> {
    let mut out = Vec::new();
    let Ok(file) = fs::File::open(path) else {
        return out;
    };
    for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(label) = line.split_whitespace().next() else {
            continue;
        };
        if let Some(rules) = parse_rule_label(label) {
            out.push(FavoriteRule {
                rules,
                label: label.to_string(),
            });
        }
    }
    out
}

/// Save favorites to a file (rewriting completely).
fn save_favorites(path: &Path, entries: &[FavoriteRule]) {
    let Ok(mut f) = fs::File::create(path) else {
        return;
    };
    let _ = f.write_all(FAVORITES_HEADER.as_bytes());
    for entry in entries {
        let _ = writeln!(f, "{}", entry.label);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_check_favorite() {
        let path = std::env::temp_dir().join("catconway_test_fav_add.txt");
        let _ = fs::remove_file(&path);

        let mut favs = Favorites::with_path(path.clone());
        let rules = Rules::conway();

        assert!(!favs.is_favorite(&rules));
        assert!(favs.add(rules));
        assert!(favs.is_favorite(&rules));
        assert_eq!(favs.entries().len(), 1);

        // Duplicate add returns false.
        assert!(!favs.add(rules));
        assert_eq!(favs.entries().len(), 1);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn remove_favorite() {
        let path = std::env::temp_dir().join("catconway_test_fav_remove.txt");
        let _ = fs::remove_file(&path);

        let mut favs = Favorites::with_path(path.clone());
        let rules = Rules::conway();

        favs.add(rules);
        assert!(favs.remove(&rules));
        assert!(!favs.is_favorite(&rules));
        assert_eq!(favs.entries().len(), 0);

        // Remove non-existent returns false.
        assert!(!favs.remove(&rules));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn toggle_favorite() {
        let path = std::env::temp_dir().join("catconway_test_fav_toggle.txt");
        let _ = fs::remove_file(&path);

        let mut favs = Favorites::with_path(path.clone());
        let rules = Rules::highlife();

        assert!(favs.toggle(&rules)); // now favorite
        assert!(favs.is_favorite(&rules));

        assert!(!favs.toggle(&rules)); // removed
        assert!(!favs.is_favorite(&rules));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn persistence_roundtrip() {
        let path = std::env::temp_dir().join("catconway_test_fav_persist.txt");
        let _ = fs::remove_file(&path);

        // Add some favorites.
        {
            let mut favs = Favorites::with_path(path.clone());
            favs.add(Rules::conway());
            favs.add(Rules::highlife());
            favs.add(Rules::seeds());
        }

        // Reload from disk.
        let favs = Favorites::with_path(path.clone());
        assert_eq!(favs.entries().len(), 3);
        assert!(favs.is_favorite(&Rules::conway()));
        assert!(favs.is_favorite(&Rules::highlife()));
        assert!(favs.is_favorite(&Rules::seeds()));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn empty_file_loads_empty() {
        let path = std::env::temp_dir().join("catconway_test_fav_empty.txt");
        let _ = fs::remove_file(&path);

        let favs = Favorites::with_path(path.clone());
        assert_eq!(favs.entries().len(), 0);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn entries_have_labels() {
        let path = std::env::temp_dir().join("catconway_test_fav_labels.txt");
        let _ = fs::remove_file(&path);

        let mut favs = Favorites::with_path(path.clone());
        favs.add(Rules::conway());
        assert_eq!(favs.entries()[0].label, "B3/S23");

        let _ = fs::remove_file(&path);
    }
}
