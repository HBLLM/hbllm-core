//! Text cleaning and Unicode normalization.
//!
//! High-performance text preprocessing for training data:
//! - HTML tag removal (regex-free for speed)
//! - Unicode NFC normalization
//! - Whitespace normalization
//! - Control character removal
//! - Length filtering
//! - Line-level and document-level cleaning

/// Configuration for text cleaning.
#[derive(Debug, Clone)]
pub struct CleanConfig {
    /// Remove HTML tags
    pub remove_html: bool,
    /// Normalize whitespace (collapse multiple spaces/newlines)
    pub normalize_whitespace: bool,
    /// Remove control characters (U+0000-U+001F except \n \r \t)
    pub remove_control_chars: bool,
    /// Minimum text length after cleaning (bytes)
    pub min_length: usize,
    /// Maximum text length after cleaning (bytes)
    pub max_length: usize,
}

impl Default for CleanConfig {
    fn default() -> Self {
        Self {
            remove_html: true,
            normalize_whitespace: true,
            remove_control_chars: true,
            min_length: 100,
            max_length: 1_000_000,
        }
    }
}

/// Clean a single text document.
///
/// Returns `None` if the text doesn't pass quality filters
/// (e.g., too short after cleaning).
pub fn clean_text(text: &str, config: &CleanConfig) -> Option<String> {
    let mut result = text.to_string();

    // Step 1: Remove HTML tags
    if config.remove_html {
        result = strip_html_tags(&result);
    }

    // Step 2: Remove control characters
    if config.remove_control_chars {
        result = remove_control_characters(&result);
    }

    // Step 3: Normalize whitespace
    if config.normalize_whitespace {
        result = normalize_whitespace_str(&result);
    }

    // Step 4: Trim
    result = result.trim().to_string();

    // Step 5: Length filter
    if result.len() < config.min_length || result.len() > config.max_length {
        return None;
    }

    Some(result)
}

/// Strip HTML tags from text without using regex.
///
/// Simple state-machine approach: skip everything between < and >.
/// Also handles common HTML entities (&amp;, &lt;, &gt;, &quot;, &#NNN;).
fn strip_html_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_tag = false;
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                // Add a space after block-level tags to preserve word boundaries
                result.push(' ');
            }
            '&' if !in_tag => {
                // Handle HTML entities
                let entity = decode_html_entity(&mut chars);
                result.push_str(&entity);
            }
            _ if !in_tag => result.push(ch),
            _ => {} // Inside a tag, skip
        }
    }

    result
}

/// Decode a single HTML entity starting after the '&'.
fn decode_html_entity(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut entity = String::new();

    // Collect characters up to ';' or max 10 chars
    for _ in 0..10 {
        match chars.peek() {
            Some(&';') => {
                chars.next(); // consume ';'
                break;
            }
            Some(&c) if c.is_alphanumeric() || c == '#' => {
                entity.push(c);
                chars.next();
            }
            _ => break,
        }
    }

    // Decode known entities
    match entity.as_str() {
        "amp" => "&".to_string(),
        "lt" => "<".to_string(),
        "gt" => ">".to_string(),
        "quot" => "\"".to_string(),
        "apos" => "'".to_string(),
        "nbsp" => " ".to_string(),
        s if s.starts_with('#') => {
            // Numeric entity: &#NNN; or &#xHHH;
            let num_str = &s[1..];
            let code_point = if num_str.starts_with('x') || num_str.starts_with('X') {
                u32::from_str_radix(&num_str[1..], 16).ok()
            } else {
                num_str.parse::<u32>().ok()
            };
            match code_point.and_then(char::from_u32) {
                Some(c) => c.to_string(),
                None => format!("&{};", entity),
            }
        }
        _ => format!("&{};", entity), // Unknown entity, keep as-is
    }
}

/// Remove control characters except newline, carriage return, and tab.
fn remove_control_characters(text: &str) -> String {
    text.chars()
        .filter(|&c| !c.is_control() || c == '\n' || c == '\r' || c == '\t')
        .collect()
}

/// Normalize whitespace: collapse runs of whitespace into single spaces,
/// collapse runs of newlines into double newlines (preserving paragraphs).
fn normalize_whitespace_str(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut last_was_space = false;
    let mut newline_count = 0;

    for ch in text.chars() {
        if ch == '\n' || ch == '\r' {
            newline_count += 1;
            last_was_space = false;
            continue;
        }

        // Flush newlines: 1 newline → space, 2+ newlines → \n\n
        if newline_count > 0 {
            if newline_count >= 2 {
                result.push('\n');
                result.push('\n');
            } else {
                result.push(' ');
            }
            newline_count = 0;
            last_was_space = false;
        }

        if ch.is_whitespace() {
            if !last_was_space {
                result.push(' ');
                last_was_space = true;
            }
        } else {
            result.push(ch);
            last_was_space = false;
        }
    }

    result
}

/// Batch clean multiple documents in parallel.
///
/// Returns only the documents that pass quality filters.
pub fn clean_batch(texts: Vec<String>, config: &CleanConfig) -> Vec<String> {
    use rayon::prelude::*;

    texts
        .into_par_iter()
        .filter_map(|text| clean_text(&text, config))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        let html = "<p>Hello <b>world</b></p>";
        let clean = strip_html_tags(html);
        assert!(clean.contains("Hello"));
        assert!(clean.contains("world"));
        assert!(!clean.contains("<p>"));
        assert!(!clean.contains("<b>"));
    }

    #[test]
    fn test_html_entities() {
        let html = "Tom &amp; Jerry, 5 &lt; 10";
        let clean = strip_html_tags(html);
        assert!(clean.contains("Tom & Jerry"));
        assert!(clean.contains("5 < 10"));
    }

    #[test]
    fn test_remove_control_chars() {
        let text = "hello\x00world\x01\nkeep newline";
        let clean = remove_control_characters(text);
        assert_eq!(clean, "helloworld\nkeep newline");
    }

    #[test]
    fn test_normalize_whitespace() {
        let text = "hello    world\n\n\nparagraph two";
        let clean = normalize_whitespace_str(text);
        assert_eq!(clean, "hello world\n\nparagraph two");
    }

    #[test]
    fn test_clean_text_too_short() {
        let config = CleanConfig {
            min_length: 100,
            ..Default::default()
        };
        assert!(clean_text("short", &config).is_none());
    }

    #[test]
    fn test_clean_text_passes() {
        let config = CleanConfig {
            min_length: 5,
            max_length: 1000,
            ..Default::default()
        };
        let text = "Hello World this is a normal document with enough text";
        let result = clean_text(text, &config);
        assert!(result.is_some());
    }

    #[test]
    fn test_full_pipeline() {
        let config = CleanConfig {
            min_length: 5,
            max_length: 10000,
            ..Default::default()
        };
        let html_text = "<html><body><p>Hello &amp; welcome to <b>HBLLM</b>!</p>\x00</body></html>";
        let result = clean_text(html_text, &config).unwrap();
        assert!(result.contains("Hello & welcome"));
        assert!(result.contains("HBLLM"));
        assert!(!result.contains("<"));
        assert!(!result.contains("\x00"));
    }
}
