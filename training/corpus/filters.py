"""Content filters for corpus cleaning.

All filters are statistical/heuristic — no hardcoded keyword blacklists.
Each filter returns (keep: bool, reason: str).
"""

import re
import math
import hashlib
from collections import Counter
from typing import Tuple, Optional


# ─── Compiled regex patterns (module-level for performance) ──────────────────

# HTML / code patterns
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")
RE_URL = re.compile(
    r"https?://[^\s<>\"']+|"
    r"www\.[^\s<>\"']+|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
RE_CSS_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)
RE_JS_BLOCK = re.compile(r"(?:function\s*\(|var\s+|let\s+|const\s+|=>\s*\{)", re.DOTALL)
RE_XML_DECL = re.compile(r"<\?xml[^?]*\?>")
RE_JSON_LIKE = re.compile(r'^\s*[\{\[][\s\S]*[\}\]]\s*$')
RE_BASE64 = re.compile(r"(?:[A-Za-z0-9+/]{40,}){2,}")
RE_MINIFIED = re.compile(r"^(?:[^a-zA-Z0-9\s]{5,}|[\w.]{80,})")

# Repeated content
RE_REPEATED_CHAR = re.compile(r"(.)\1{5,}")
RE_REPEATED_WORD = re.compile(r"\b(\w+)(\s+\1){3,}\b", re.IGNORECASE)
RE_REPEATED_LINE = re.compile(r"^(.+)$", re.MULTILINE)

# OCR garbage patterns
RE_OCR_GARBAGE = re.compile(r"[^\x00-\x7F]{10,}")
RE_MOJIBAKE = re.compile(r"(?:Ã©|Ã¨|Ã¡|Ã ','Ã¶|Ã¼|Ã®|Ã«|Ã£|Ãµ)")

# Link farm / SEO patterns
RE_LINK_FARM = re.compile(r"(?:click here|buy now|visit us|order now|limited time|act now)", re.IGNORECASE)
RE_KEYWORD_STUFFING = re.compile(r"\b(\w+)\b.*\b\1\b.*\b\1\b.*\b\1\b", re.IGNOREILINE if False else re.DOTALL)


# ─── Filter base class ──────────────────────────────────────────────────────

class ContentFilter:
    """Base class for content filters."""

    name: str = "base"
    description: str = ""

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        raise NotImplementedError


# ─── Adult / NSFW content filter ────────────────────────────────────────────

class NSFWFilter(ContentFilter):
    """Detect adult/NSFW content using statistical signals.

    Uses pattern density analysis rather than keyword blacklists.
    Signals: sexual content density, escort ad patterns, explicit service keywords.
    """

    name = "nsfw"
    description = "Adult/NSFW content detection"

    # Statistical patterns (not exact keywords)
    SEXUAL_PATTERNS = re.compile(
        r"(?:"
        r"escort|incall|outcall|brothel|hookup|hook up|one night stand|"
        r"fuck|sex|nude|naked|porn|xxx|erotica|erotic|"
        r"blowjob|handjob|cumshot|gangbang|threesome|"
        r"strip[ps]|lapdance|stripper|"
        r"adult dating|sugar daddy|sugar baby|"
        r"live cam|webcam girl|cam show|"
        r"sex toy|vibrator|dildo"
        r")",
        re.IGNORECASE
    )

    EXPLICIT_SERVICE_PATTERNS = re.compile(
        r"(?:"
        r"available for (?:incall|outcall)|"
        r"call(?:s)? (?:now|me)|"
        r"discreet (?:encounter|meeting|service)|"
        r"no strings attached|"
        r"personal (?:ads?|services?)|"
        r"body rub|happy ending|"
        r"fully nude|full service"
        r")",
        re.IGNORECASE
    )

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 20:
            return True, ""

        # Calculate sexual content density
        sexual_matches = len(self.SEXUAL_PATTERNS.findall(text))
        service_matches = len(self.EXPLICIT_SERVICE_PATTERNS.findall(text))
        words = len(text.split())

        if words == 0:
            return True, ""

        density = (sexual_matches + service_matches * 2) / words

        if density > 0.03:  # >3% sexual content density
            return False, f"nsfw_density={density:.3f}"

        # Check for escort ad structure (short text, high keyword density)
        if words < 100 and (sexual_matches + service_matches) > 3:
            return False, f"nsfwEscortAd={sexual_matches + service_matches}"

        return True, ""


# ─── Gambling / Casino spam filter ──────────────────────────────────────────

class GamblingFilter(ContentFilter):
    """Detect gambling/casino spam using pattern density."""

    name = "gambling"
    description = "Gambling/casino spam detection"

    GAMBLING_PATTERNS = re.compile(
        r"(?:"
        r"casino|poker|blackjack|roulette|slots|jackpot|"
        r"bet(?:ting|s)?|wager|gambl|"
        r"online casino|live casino|mobile casino|"
        r"free spins|bonus code|deposit bonus|"
        r"play (?:now|free|for real)|"
        r"real money (?:casino|slots|poker)|"
        r"win (?:big|real|cash|money)"
        r")",
        re.IGNORECASE
    )

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 20:
            return True, ""

        matches = len(self.GAMBLING_PATTERNS.findall(text))
        words = len(text.split())

        if words == 0:
            return True, ""

        density = matches / words

        if density > 0.02:
            return False, f"gambling_density={density:.3f}"

        return True, ""


# ─── SEO spam filter ────────────────────────────────────────────────────────

class SEOSpamFilter(ContentFilter):
    """Detect SEO spam, keyword stuffing, and link farms.

    Uses statistical signals: word repetition, link density, keyword stuffing.
    """

    name = "seo_spam"
    description = "SEO spam and keyword stuffing detection"

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 30:
            return True, ""

        words = text.split()
        word_count = len(words)

        if word_count < 5:
            return True, ""

        # Check for excessive URL density
        urls = len(RE_URL.findall(text))
        url_density = urls / word_count
        if url_density > 0.1:  # >10% URLs
            return False, f"url_density={url_density:.3f}"

        # Check for repeated words (keyword stuffing)
        word_freq = Counter(w.lower() for w in words if len(w) > 3)
        if word_freq:
            max_freq = max(word_freq.values())
            max_freq_ratio = max_freq / word_count
            if max_freq_ratio > 0.08:  # Single word >8% of text
                return False, f"keyword_stuffing={max_freq_ratio:.3f}"

        # Check for link farm patterns
        link_farm_matches = len(RE_LINK_FARM.findall(text))
        if link_farm_matches > 3:
            return False, f"link_farm={link_farm_matches}"

        # Check for excessive capitalization (shouting)
        caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        caps_ratio = caps_words / word_count
        if caps_ratio > 0.3 and word_count > 10:
            return False, f"caps_ratio={caps_ratio:.3f}"

        return True, ""


# ─── HTML / code blob filter ────────────────────────────────────────────────

class HTMLCodeFilter(ContentFilter):
    """Detect HTML boilerplate, JavaScript, CSS, XML, JSON, minified code, base64."""

    name = "html_code"
    description = "HTML, JavaScript, CSS, XML, JSON, minified code, base64 detection"

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 10:
            return True, ""

        # Check for XML declarations
        if RE_XML_DECL.match(text.strip()):
            return False, "xml_declaration"

        # Check for JSON blobs
        stripped = text.strip()
        if RE_JSON_LIKE.match(stripped) and len(stripped) > 100:
            # Verify it's actually JSON-like
            if stripped[0] in '{[' and stripped[-1] in '}]':
                return False, "json_blob"

        # Check for base64 blobs
        if RE_BASE64.search(text):
            return False, "base64_blob"

        # Calculate HTML tag density
        html_tags = len(RE_HTML_TAG.findall(text))
        words = len(text.split())
        if words > 0:
            html_density = html_tags / words
            if html_density > 0.3:  # >30% HTML tags
                return False, f"html_density={html_density:.3f}"

        # Check for CSS blocks
        css_matches = len(RE_CSS_BLOCK.findall(text))
        if css_matches > 5:
            return False, f"css_blocks={css_matches}"

        # Check for JavaScript
        js_matches = len(RE_JS_BLOCK.findall(text))
        if js_matches > 3:
            return False, f"javascript={js_matches}"

        # Check for minified code (very long lines with few spaces)
        lines = text.split('\n')
        for line in lines:
            if len(line) > 500:
                spaces = line.count(' ')
                if spaces < 10:
                    return False, "minified_code"

        return True, ""


# ─── Repeated content filter ────────────────────────────────────────────────

class RepeatedContentFilter(ContentFilter):
    """Detect repeated characters, words, and lines."""

    name = "repeated_content"
    description = "Repeated characters, words, and lines detection"

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 10:
            return True, ""

        # Repeated characters (e.g., "aaaaaaa", "!!!!!!!!")
        if RE_REPEATED_CHAR.search(text):
            return False, "repeated_characters"

        # Repeated words (e.g., "the the the the")
        repeated_words = len(RE_REPEATED_WORD.findall(text))
        if repeated_words > 0:
            return False, f"repeated_words={repeated_words}"

        # Repeated lines
        lines = RE_REPEATED_LINE.findall(text)
        if len(lines) > 3:
            line_counts = Counter(lines)
            max重复 = max(line_counts.values())
            if max重复 > len(lines) * 0.3:  # >30% duplicate lines
                return False, f"repeated_lines={max重复}"

        return True, ""


# ─── Low language quality filter ────────────────────────────────────────────

class LanguageQualityFilter(ContentFilter):
    """Detect low-quality text using statistical signals.

    Checks: alphabetic ratio, URL density, punctuation density, character entropy.
    """

    name = "language_quality"
    description = "Low language quality detection"

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 20:
            return True, ""

        # Alphabetic ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / len(text)
        if alpha_ratio < 0.5:  # <50% alphabetic characters
            return False, f"low_alpha_ratio={alpha_ratio:.3f}"

        # URL density
        urls = len(RE_URL.findall(text))
        words = len(text.split())
        if words > 0:
            url_density = urls / words
            if url_density > 0.15:  # >15% URLs
                return False, f"high_url_density={url_density:.3f}"

        # Punctuation density
        punct_count = sum(1 for c in text if c in '.,!?;:()-[]{}"\'')
        punct_ratio = punct_count / len(text)
        if punct_ratio > 0.4:  # >40% punctuation
            return False, f"high_punct_ratio={punct_ratio:.3f}"

        # Character entropy (very low = repetitive/low quality)
        char_freq = Counter(text)
        total = len(text)
        entropy = -sum((c / total) * math.log2(c / total) for c in char_freq.values())
        if entropy < 3.0:  # Very low entropy
            return False, f"low_entropy={entropy:.2f}"

        # OCR garbage detection
        non_ascii_runs = len(RE_OCR_GARBAGE.findall(text))
        if non_ascii_runs > 5:
            return False, f"ocr_garbage={non_ascii_runs}"

        # Mojibake detection
        if RE_MOJIBAKE.search(text):
            return False, "mojibake"

        return True, ""


# ─── OCR garbage filter ─────────────────────────────────────────────────────

class OCRGarbageFilter(ContentFilter):
    """Detect OCR artifacts and garbled text."""

    name = "ocr_garbage"
    description = "OCR artifact detection"

    # Common OCR errors
    OCR_PATTERNS = re.compile(
        r"(?:"
        r"[|}{]\s*[|}{]|"           # Broken table borders
        r"(?:^|\s)[Il1|](?:$|\s)|"  # Isolated I/l/1/|
        r"â€™|Ã©|Ã¨|Ã¡|Ã ','Ã¶|Ã¼|Ã®|Ã«|Ã£|Ãµ|"  # Mojibake
        r"â€[cx]|â€œ|â€?"           # Smart quote artifacts
        r")",
        re.MULTILINE
    )

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 20:
            return True, ""

        matches = len(self.OCR_PATTERNS.findall(text))
        words = len(text.split())

        if words > 0 and matches / words > 0.05:  # >5% OCR artifacts
            return False, f"ocr_artifacts={matches}"

        return True, ""


# ─── Document length filter ─────────────────────────────────────────────────

class DocumentLengthFilter(ContentFilter):
    """Filter by document length. Split long documents instead of discarding."""

    name = "document_length"
    description = "Document length filtering with chunking"

    def __init__(self, min_length: int = 50, max_length: int = 100000,
                 split_long: bool = True, split_max: int = 10000):
        self.min_length = min_length
        self.max_length = max_length
        self.split_long = split_long
        self.split_max = split_max

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text:
            return False, "empty_document"

        length = len(text)

        if length < self.min_length:
            return False, f"too_short={length}"

        if length > self.max_length and not self.split_long:
            return False, f"too_long={length}"

        return True, ""

    def split_document(self, text: str) -> list:
        """Split long document into reasonable chunks."""
        if len(text) <= self.split_max:
            return [text]

        chunks = []
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > self.split_max:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        # If still too long, split by sentences
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.split_max:
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) + 1 > self.split_max:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        sub_chunk = sent
                    else:
                        sub_chunk = sub_chunk + " " + sent if sub_chunk else sent
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks if final_chunks else [text]


# ─── Unicode quality filter ─────────────────────────────────────────────────

class UnicodeQualityFilter(ContentFilter):
    """Filter documents with malformed or problematic Unicode."""

    name = "unicode_quality"
    description = "Malformed Unicode detection"

    # Control characters (except tab, newline, carriage return)
    CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text:
            return True, ""

        # Check for control characters
        control_matches = len(self.CONTROL_CHARS.findall(text))
        if control_matches > 0:
            return False, f"control_chars={control_matches}"

        # Check for replacement characters
        if '\ufffd' in text:
            return False, "replacement_chars"

        # Check for excessive non-BMP characters
        non_bmp = sum(1 for c in text if ord(c) > 0xFFFF)
        if non_bmp > 10:
            return False, f"non_bmp_chars={non_bmp}"

        return True, ""


# ─── Language detection filter ──────────────────────────────────────────────

class LanguageFilter(ContentFilter):
    """Filter non-English documents using statistical heuristics.

    Uses character frequency analysis rather than external libraries.
    """

    name = "language"
    description = "Non-English content detection"

    # Common English word frequencies (top 100)
    COMMON_ENGLISH = frozenset([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see",
        "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work",
        "first", "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "are", "was", "were", "been", "being",
    ])

    def __call__(self, text: str, metadata: Optional[dict] = None) -> Tuple[bool, str]:
        if not text or len(text) < 50:
            return True, ""

        words = text.lower().split()
        if len(words) < 10:
            return True, ""

        # Check English word coverage
        english_words = sum(1 for w in words if w.strip('.,!?;:()') in self.COMMON_ENGLISH)
        english_ratio = english_words / len(words)

        if english_ratio < 0.15:  # <15% common English words
            return False, f"non_english={english_ratio:.3f}"

        # Check Latin character ratio (simple heuristic)
        latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        latin_ratio = latin_chars / len(text)

        if latin_ratio < 0.5:  # <50% Latin characters
            return False, f"non_latin={latin_ratio:.3f}"

        return True, ""


# ─── All filters registry ──────────────────────────────────────────────────

ALL_FILTERS = {
    "nsfw": NSFWFilter,
    "gambling": GamblingFilter,
    "seo_spam": SEOSpamFilter,
    "html_code": HTMLCodeFilter,
    "repeated_content": RepeatedContentFilter,
    "language_quality": LanguageQualityFilter,
    "ocr_garbage": OCRGarbageFilter,
    "document_length": DocumentLengthFilter,
    "unicode_quality": UnicodeQualityFilter,
    "language": LanguageFilter,
}


def get_filter(name: str, **kwargs) -> ContentFilter:
    """Get a filter by name."""
    if name not in ALL_FILTERS:
        raise ValueError(f"Unknown filter: {name}. Available: {list(ALL_FILTERS.keys())}")
    return ALL_FILTERS[name](**kwargs)


def build_filter_chain(config: dict) -> list:
    """Build a filter chain from config.

    Config format:
        filters:
          nsfw: {enabled: true}
          gambling: {enabled: true}
          document_length: {enabled: true, min_length: 50, max_length: 100000}
    """
    chain = []
    filters_config = config.get("filters", {})

    for name, filter_config in filters_config.items():
        if not filter_config.get("enabled", True):
            continue

        kwargs = {k: v for k, v in filter_config.items() if k != "enabled"}
        try:
            filter_obj = get_filter(name, **kwargs)
            chain.append(filter_obj)
        except Exception as e:
            print(f"Warning: Failed to create filter '{name}': {e}")

    return chain
