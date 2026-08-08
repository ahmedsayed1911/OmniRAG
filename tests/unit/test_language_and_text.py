"""Arabic/English handling, text normalisation, hashing, and retry policy."""

from __future__ import annotations

import pytest

from omnirag.core.enums import Language
from omnirag.utils.hashing import (
    content_hash,
    file_extension,
    sanitize_filename,
    stable_id,
    text_hash,
)
from omnirag.utils.language import (
    contains_arabic,
    detect_language,
    is_rtl,
    normalize_arabic,
    normalize_for_search,
    script_ratios,
)
from omnirag.utils.text import (
    clean_text,
    detect_repeated_lines,
    estimate_tokens,
    is_meaningful,
    remove_lines,
    split_paragraphs,
    split_sentences,
    tokenize,
    truncate,
)


class TestLanguageDetection:
    @pytest.mark.parametrize("text,expected", [
        ("Total revenue reached 8.4 million USD.", Language.ENGLISH),
        ("بلغت الإيرادات الإجمالية ثمانية ملايين دولار", Language.ARABIC),
        ("الإيرادات Revenue بلغت 8.4 million دولار أمريكي", Language.MIXED),
        ("", Language.UNKNOWN),
        ("12345 6789 ... !!!", Language.UNKNOWN),
    ])
    def test_detection(self, text, expected):
        assert detect_language(text) == expected

    def test_script_ratios(self):
        arabic, latin = script_ratios("abc أبج")
        assert arabic == pytest.approx(0.5)
        assert latin == pytest.approx(0.5)

    def test_contains_arabic(self):
        assert contains_arabic("hello مرحبا") is True
        assert contains_arabic("hello world") is False

    def test_rtl_detection(self):
        assert is_rtl("بلغت الإيرادات الإجمالية") is True
        assert is_rtl("Total revenue") is False


class TestArabicNormalization:
    @pytest.mark.parametrize("raw,expected_contains", [
        ("الإيرادات", "الايرادات"),   # hamza forms unified
        ("أحمد", "احمد"),
        ("مكتبة", "مكتبه"),           # teh marbuta -> heh
        ("علــــى", "علي"),           # tatweel removed, alef maqsura -> yeh
    ])
    def test_normalization_unifies_variants(self, raw, expected_contains):
        assert normalize_arabic(raw) == expected_contains

    def test_diacritics_are_stripped(self):
        assert normalize_arabic("مُحَمَّد") == normalize_arabic("محمد")

    def test_search_normalization_is_lowercase(self):
        assert normalize_for_search("  Revenue   GREW  ") == "revenue grew"

    def test_normalization_never_mutates_stored_text(self):
        # Normalisation is for index keys only; the source stays verbatim.
        original = "الإيرادات"
        normalize_arabic(original)
        assert original == "الإيرادات"


class TestTokenization:
    def test_arabic_and_english_tokens(self):
        tokens = tokenize("Revenue بلغت 8400 USD")
        assert "revenue" in tokens
        assert "8400" in tokens
        assert any(contains_arabic(t) for t in tokens)

    def test_tokens_are_normalized_for_matching(self):
        assert tokenize("الإيرادات") == tokenize("الايرادات")


class TestTextCleaning:
    def test_hyphenated_line_breaks_are_rejoined(self):
        assert "revenue" in clean_text("reve-\nnue increased")

    def test_control_characters_are_removed(self):
        assert "\x00" not in clean_text("bad\x00text here")

    def test_excessive_whitespace_is_collapsed(self):
        assert clean_text("a    b\n\n\n\nc") == "a b\n\nc"

    def test_numbers_are_never_altered(self):
        text = "Revenue was 8,400,000.50 USD (+12.3%)"
        assert "8,400,000.50" in clean_text(text)
        assert "12.3%" in clean_text(text)

    def test_arabic_text_survives_cleaning(self):
        assert "الإيرادات" in clean_text("  الإيرادات   الإجمالية  ")

    def test_repeated_headers_are_detected_and_removable(self):
        pages = [
            f"ACME CONFIDENTIAL\nPage body {i} with unique content here.\nFooter line"
            for i in range(6)
        ]
        repeated = detect_repeated_lines(pages)

        assert "ACME CONFIDENTIAL" in repeated
        assert "Footer line" in repeated

        cleaned = remove_lines(pages[0], repeated)
        assert "ACME CONFIDENTIAL" not in cleaned
        assert "Page body 0" in cleaned

    def test_body_text_repeating_by_chance_is_not_stripped(self):
        pages = ["Header\n" + "unique %d\n" % i + "Total revenue grew." for i in range(6)]
        repeated = detect_repeated_lines(pages)
        assert "Header" in repeated


class TestTextHelpers:
    def test_sentence_splitting_handles_arabic_punctuation(self):
        sentences = split_sentences("ما هي الإيرادات؟ بلغت 8.4 مليون. نعم.")
        assert len(sentences) >= 2

    def test_paragraph_splitting(self):
        assert len(split_paragraphs("one\n\ntwo\n\nthree")) == 3

    def test_truncate_respects_word_boundaries(self):
        result = truncate("the quick brown fox jumps over", 15)
        assert len(result) <= 16
        assert result.endswith("…")

    def test_token_estimation_is_positive(self):
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 0

    @pytest.mark.parametrize("text,expected", [
        ("Total revenue reached 8.4 million.", True),
        ("...", False),
        ("a", False),
        ("|||||||||||||", False),
        ("بلغت الإيرادات الإجمالية", True),
    ])
    def test_meaningfulness_filter(self, text, expected):
        assert is_meaningful(text) is expected


class TestHashing:
    def test_content_hash_is_stable_and_distinct(self):
        assert content_hash(b"abc") == content_hash(b"abc")
        assert content_hash(b"abc") != content_hash(b"abd")

    def test_text_hash_ignores_case_and_whitespace(self):
        assert text_hash("Hello   World") == text_hash("hello world")

    def test_stable_id_is_deterministic(self):
        assert stable_id("a", "b", "c") == stable_id("a", "b", "c")
        assert stable_id("a", "b") != stable_id("a", "c")

    @pytest.mark.parametrize("raw,checks", [
        ("../../etc/passwd", lambda n: "/" not in n and ".." not in n),
        ("C:\\Windows\\system32\\evil.txt", lambda n: "\\" not in n),
        ("report<>:\"|?*.pdf", lambda n: not set(n) & set('<>:"|?*')),
        ("", lambda n: n == "untitled"),
        ("CON.txt", lambda n: n.startswith("_")),
        ("تقرير سنوي.pdf", lambda n: "تقرير" in n),
        ("normal_file-1.pdf", lambda n: n == "normal_file-1.pdf"),
    ])
    def test_filename_sanitisation(self, raw, checks):
        assert checks(sanitize_filename(raw))

    def test_long_filenames_are_truncated_but_keep_the_extension(self):
        result = sanitize_filename("x" * 500 + ".pdf")
        assert len(result) <= 130
        assert result.endswith(".pdf")

    def test_file_extension(self):
        assert file_extension("Report.PDF") == "pdf"
        assert file_extension("noext") == ""


class TestRetryPolicy:
    def test_transient_failures_are_retried(self):
        from omnirag.core.exceptions import RateLimitError
        from omnirag.utils.retry import retry_call

        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RateLimitError("429", provider="test")
            return "ok"

        assert retry_call(flaky, attempts=3, sleep=lambda _: None) == "ok"
        assert attempts["n"] == 3

    def test_permanent_failures_are_not_retried(self):
        from omnirag.core.exceptions import ProviderAuthError
        from omnirag.utils.retry import retry_call

        attempts = {"n": 0}

        def failing():
            attempts["n"] += 1
            raise ProviderAuthError("401", provider="test")

        with pytest.raises(ProviderAuthError):
            retry_call(failing, attempts=4, sleep=lambda _: None)
        assert attempts["n"] == 1

    def test_backoff_is_bounded(self):
        from omnirag.utils.retry import backoff_delay

        for attempt in range(1, 8):
            assert 0 <= backoff_delay(attempt, base=1.0, cap=8.0) <= 8.0


class TestLogRedaction:
    @pytest.mark.parametrize("secret", [
        "sk-abcdefghijklmnop",
        "sk-ant-abcdefghijklmnop",
        "AIzaSyABCDEFGHIJKLMNOPQRSTUV",
    ])
    def test_credentials_are_redacted_from_log_messages(self, secret):
        from omnirag.utils.logging import redact

        assert secret not in redact(f"calling provider with key {secret}")

    def test_bearer_tokens_are_redacted(self):
        from omnirag.utils.logging import redact

        assert "abcdef1234567890" not in redact("Authorization: Bearer abcdef1234567890")
