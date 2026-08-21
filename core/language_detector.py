"""
Silent-Language Memory — automatic language detection for Jarvis.

JARVIS 6.3 — Detects the language of the user's *first* spoken utterance
without sending text to any external API.  Uses a lightweight two-stage
model: word-level function-word matching (primary) + character-trigram
fallback (for short or ambiguous utterances).

After detection the result is stored in the user's memory profile under
``identity.language`` so every subsequent run uses the persisted language.

Example::

    from core.language_detector import LanguageDetector
    ld = LanguageDetector()
    code = ld.detect("Hola, ¿cómo estás hoy?")
    code            # 'es'
    ld.is_reliable  # True when confidence exceeds the threshold
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Stage 1 — Word-level language indicators (function words, most distinctive)
# All words are stored WITHOUT diacritics — the tokenizer strips accents.
# --------------------------------------------------------------------------- #

_LANGUAGE_WORDS: dict[str, set[str]] = {
    "en": {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "can", "may", "might", "must", "shall", "of", "in", "on",
        "at", "to", "for", "with", "by", "from", "as", "into", "about",
        "how", "what", "when", "where", "why", "which", "who", "whom",
        "this", "that", "these", "those", "and", "or", "but", "not",
        "no", "yes", "please", "thank", "thanks", "hello", "hi", "today",
        "tomorrow", "yesterday", "morning", "afternoon", "evening",
        "good", "bad", "goodbye", "bye", "see", "go", "come",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "es", "son",
        "esta", "estan", "fue", "fueron", "ser", "estar", "tener",
        "tengo", "tienes", "tiene", "tenemos", "tienen", "de", "del", "en",
        "por", "para", "con", "y", "o", "pero", "no", "si", "muy",
        "hoy", "manana", "hola", "gracias", "como", "que", "quien",
        "donde", "cuando", "porque", "mas", "menos", "algo", "este", "estos",
        "estas", "esa", "ese", "esos", "esas", "mi", "tu", "su",
        "voy", "vas", "va", "vamos", "vais", "van", "soy", "eres",
        "bien", "mal", "hacer", "decir", "dar", "ir", "ver", "saber",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "du", "de", "del", "est",
        "sont", "etais", "etaient", "etre", "avoir", "je", "tu",
        "il", "elle", "nous", "vous", "ils", "elles", "dans", "sur", "pour",
        "par", "avec", "et", "ou", "mais", "pas", "plus", "moins", "tres",
        "oui", "non", "merci", "bonjour", "aujourd'hui", "demain",
        "comment", "quoi", "ou", "quand", "pourquoi", "quel", "quelle",
        "qui", "aussi", "encore", "peu", "tout", "toute", "tous",
        "maison", "temps", "jour", "nuit", "homme", "femme",
        "bon", "bonne", "mauvais", "mauvaise", "grand", "grande",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "einer", "eines", "einem",
        "einen", "ist", "sind", "war", "waren", "sein", "seine", "haben",
        "hat", "hatten", "habe", "ich", "du", "er", "sie", "es", "wir",
        "ihr", "in", "auf", "fur", "mit", "und", "oder", "aber",
        "nicht", "von", "zu", "den", "dem", "des", "aus", "bei", "nach",
        "vor", "ja", "nein", "danke", "hallo", "guten", "gute", "tag",
        "morgen", "abend", "mittag", "wie", "wo", "wann", "warum", "was",
        "welche", "wer", "wieviel", "sehr", "auch", "noch", "alle", "viel",
        "kein", "keine", "keinen", "ihre", "mein", "meine", "dein", "deine",
        "machen", "finden", "geben", "kommen", "gehen", "sehen", "lassen",
    },
    "it": {
        "il", "la", "i", "gli", "le", "uno", "una", "un", "e", "o", "ma",
        "non", "piu", "meno", "molto", "si", "no", "grazie", "ciao",
        "come", "cosa", "quando", "dove", "perche", "anche", "ancora",
        "bene", "male", "ieri", "oggi", "domani", "mattina", "sera",
        "notte", "giorno", "anno", "mese", "settimana", "ora", "uomo",
        "donna", "casa", "tempo", "vita", "mondo", "fare", "dire",
        "dare", "andare", "vedere", "potere", "dovere", "sapere",
        "venire", "trovare", "mettere", "stare", "stai", "sono",
        "era", "erano", "essere", "avere", "ho", "hai", "ha", "hanno",
        "di", "del", "della", "dei", "delle", "su", "per", "très",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "e", "ou", "mas",
        "nao", "mais", "menos", "muito", "sim", "obrigado", "ola", "como",
        "onde", "quando", "porque", "qual", "quem", "quanto",
        "tambem", "ainda", "hoje", "amanha", "ontem",
        "dia", "noite", "ano", "semana", "vida", "tempo",
        "pessoa", "casa", "mundo", "homem", "mulher", "fazer", "falar",
        "dar", "ir", "haver", "todo", "toda", "outro", "outra",
        "este", "esta", "estes", "estas", "essa", "esse", "estou",
        "estou", "sao", "esta", "estava", "fui", "foste", "foi",
    },
    "nl": {
        "de", "het", "een", "twee", "drie", "vier", "is", "zijn", "was",
        "waren", "hebben", "hebt", "zal", "zouden", "zou", "kan",
        "in", "op", "van", "en", "of", "maar", "niet", "ja", "nee",
        "dank", "hallo", "hoe", "waar", "wanneer", "waarom", "wat",
        "wie", "welke", "hoeveel", "ook", "nog", "alle", "mijn",
        "jij", "hij", "zij", "wij", "hun", "ik", "jouw", "ons",
        "goedemorgen", "goedenmiddag", "goedenavond",
    },
    "ru": {
        "и", "в", "не", "что", "он", "на", "я", "с", "со", "как", "а",
        "все", "она", "так", "его", "есть", "ее", "или", "быть",
        "да", "ты", "к", "из", "у", "же", "вы", "за", "по", "только",
        "мне", "бы", "себя", "нет", "если", "еще", "очень", "совсем",
    },
}

# Prefix stems for partial matching on out-of-vocabulary words. Stem length >= 3.
_LANGUAGE_PREFIXES: dict[str, set[str]] = {
    "en": {"the", "ing", "ion", "ver", "wit", "out", "pre", "sub"},
    "es": {"est", "hab", "com", "pod", "pued", "dir", "cual", "much"},
    "fr": {"par", "sou", "tou", "com", "pou", "vou", "ser", "ett", "ment"},
    "de": {"geh", "kom", "spre", "mac", "find", "seh", "hal", "bring", "lich"},
    "it": {"com", "qu", "pot", "stell", "mand", "cher", "dir"},
    "pt": {"com", "est", "fala", "pod", "sao", "tend", "mun", "cresc"},
    "nl": {"wer", "zij", "hun", "dat", "die", "mijn", "jouw"},
    "ru": {"что", "как", "где", "кто", "мне", "ник", "все"},
}

# Prefix stems for partial matching on out-of-vocabulary words. Stem length >= 3.
_LANGUAGE_PREFIXES: dict[str, set[str]] = {
    "en": {"the", "ing", "ion", "ver", "wit", "out", "pre", "sub"},
    "es": {"est", "hab", "com", "pod", "pued", "dir", "cual", "much"},
    "fr": {"par", "sou", "tou", "com", "pou", "vou", "ser", "ett", "ment"},
    "de": {"geh", "kom", "spre", "mac", "find", "seh", "hal", "bring", "lich"},
    "it": {"com", "qu", "pot", "stell", "mand", "cher", "dir"},
    "pt": {"com", "est", "fala", "pod", "sao", "tend", "mun", "cresc"},
    "nl": {"wer", "zij", "hun", "dat", "die", "mijn", "jouw"},
    "ru": {"что", "как", "где", "кто", "мне", "ник", "все"},
}

# Map ISO 639-1 codes → human-readable names (used in system prompts / TTS config)
_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "bn": "Bangla",
}


# --------------------------------------------------------------------------- #
# Stage 2 — Character-trigram fallback (used when word matching is weak)
# --------------------------------------------------------------------------- #

# High-frequency trigrams per language (from public-domain corpora).
# Scores represent relative frequency weight.
_TRIGRAM_WEIGHTS: dict[str, dict[str, float]] = {
    "en": {
        "the": 2.3, "and": 1.8, "ing": 1.7, "ion": 1.4, "for": 0.9,
        " th": 1.1, "he ": 1.2, "ver": 0.6, "wit": 0.5, " on": 0.7,
    },
    "es": {
        "que": 2.5, "cion": 2.0, "dad": 1.2, "est": 0.9,
        " de": 1.3, "la ": 1.0, "los": 0.8, "ras": 0.7, "ion": 1.5,
    },
    "fr": {
        "ent": 1.9, "ant": 1.7, "les": 1.6, " que": 1.4, " de": 1.3,
        "ont": 1.1, "pas": 1.0, "men": 0.9, "tio": 0.8, "eur": 0.7,
    },
    "de": {
        "und": 1.8, "den": 1.5, "ten": 1.4, "cht": 1.3, "schen": 1.1,
        "en ": 0.8, "ge ": 0.7, "ung": 0.6, "gen": 0.9,
    },
    "it": {
        "che": 2.2, "cia": 1.8, "ion": 1.6, "one": 1.5, "ere": 1.3,
        "sta": 1.1, "que": 0.8, "zio": 0.5, "oni": 0.4, "nte": 1.0,
    },
    "pt": {
        "que": 2.0, "cao": 1.8, "dad": 1.1, "oes": 0.9,
        " est": 0.8, "com": 0.7, "pro": 0.6, "sao": 0.6,
    },
    "nl": {
        "de ": 1.5, "en ": 1.4, "den": 1.2, "van": 1.0, "het": 1.1,
        "een": 0.9, "ing": 0.8, "nde": 0.7, "sch": 0.6, "ijd": 0.5,
    },
    "ru": {
        "ost": 1.5, "yy ": 1.2, "ie ": 0.9, "tsc": 0.7,
        "ova": 0.4, "sta": 0.5, "est": 0.3,
    },
    "bn": {
        "আমি": 2.0, "আছে": 1.8, "কেমন": 1.6, "হবে": 1.5, "কোথা": 1.4,
        "কেন": 1.3, "কি": 1.2, "নাই": 1.1, "করা": 1.0, "হয়": 0.9,
    },
}


# Unicode script ranges for CJK + Bengali detection
_CJK_RANGE = range(0x4E00, 0x9FFF)
_HIRAGANA_RANGE = range(0x3040, 0x309F)
_KATAKANA_RANGE = range(0x30A0, 0x30FF)
_HANGUL_RANGE = range(0xAC00, 0xD7AF)
_BENGALI_RANGE = range(0x0980, 0x09FF)


@dataclass
class DetectionResult:
    """Result of a language detection attempt."""

    language: str            # ISO 639-1 code (e.g. "en", "es")
    confidence: float        # 0.0 – 1.0
    is_reliable: bool        # True if confidence >= threshold

    def __repr__(self) -> str:
        return (
            f"DetectionResult(language={self.language!r}, "
            f"confidence={self.confidence:.2f}, reliable={self.is_reliable})"
        )


class LanguageDetector:
    """Detect language from text using word + trigram models.

    No external dependencies — all language indicator sets are compiled in.
    """

    def __init__(self, confidence_threshold: float = 0.4):
        self._confidence_threshold = confidence_threshold
        self.is_reliable = False

    @staticmethod
    def _has_cjk(text: str) -> tuple[bool, bool, bool, bool, bool]:
        """Return (has_cjk, has_hiragana, has_katakana, has_hangul, has_bengali)."""
        has_cjk = has_hira = has_kata = has_hangul = has_bengali = False
        for ch in text:
            cp = ord(ch)
            if cp in _CJK_RANGE:
                has_cjk = True
            elif cp in _HIRAGANA_RANGE:
                has_hira = True
            elif cp in _KATAKANA_RANGE:
                has_kata = True
            elif cp in _HANGUL_RANGE:
                has_hangul = True
            elif cp in _BENGALI_RANGE:
                has_bengali = True
        return has_cjk, has_hira, has_kata, has_hangul, has_bengali

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip accents/digits/punct, return word tokens."""
        import unicodedata
        text = text.lower().strip()
        # Normalize: remove diacritics
        nfkd = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in nfkd if not unicodedata.combining(c))
        text = re.sub(r"\d+", " ", text)
        tokens = re.findall(r"[a-z]+|[\u0980-\u09FF]+", text, flags=re.UNICODE)
        return [t for t in tokens if len(t) >= 1]

    def _word_score(self, tokens: list[str]) -> dict[str, float]:
        """Score each language by word-token matches.

        Only tokens that are *unique* to a single language contribute to the
        score, so ambiguous words (e.g. "come" in English+Italian) don't
        create false positives.
        """
        # Build token→languages map
        token_langs: dict[str, set[str]] = defaultdict(set)
        for lang, words in _LANGUAGE_WORDS.items():
            for w in words:
                token_langs[w].add(lang)

        scores: dict[str, float] = defaultdict(float)
        for tok in tokens:
            langs = token_langs.get(tok, set())
            if len(langs) == 1:
                # Unique to one language — strong signal
                scores[next(iter(langs))] += 2
            elif len(langs) > 1:
                # Ambiguous — skip (no discriminating power)
                pass
            else:
                # Not in any word list; try prefix matching
                for lang, prefixes in _LANGUAGE_PREFIXES.items():
                    for stem in prefixes:
                        if tok.startswith(stem) and len(stem) >= 3:
                            scores[lang] += 0.5
                            break
        return {k: v for k, v in scores.items() if v > 0}

    def _trigram_score(self, text: str) -> dict[str, float]:
        """Score each language by character-trigram matches."""
        text = text.lower()
        # Build trigrams including CJK characters
        trigrams = [text[i:i + 3] for i in range(len(text) - 2)]
        if len(trigrams) < 2:
            return {}

        scores: dict[str, float] = defaultdict(float)
        for gram in trigrams:
            for lang, table in _TRIGRAM_WEIGHTS.items():
                w = table.get(gram, 0.0)
                if w > 0:
                    scores[lang] += w
        return dict(scores)

    def detect(self, text: str) -> Optional[str]:
        """Detect the language of *text* and return the ISO 639-1 code.

        Returns ``None`` if the text is too short or uncertain.
        """
        if not text or len(text.strip()) < 3:
            return None

        # Stage 1 — CJK / Japanese / Bengali script detection (unambiguous)
        has_cjk, has_hira, has_kata, has_hangul, has_bengali = self._has_cjk(text)
        if has_hira:
            self.is_reliable = True
            return "ja"
        if has_hangul:
            self.is_reliable = True
            return "ko"
        if has_cjk:
            self.is_reliable = True
            return "zh"
        if has_bengali:
            self.is_reliable = True
            return "bn"

        tokens = self._tokenize(text)
        if len(tokens) < 1:
            return None

        # Stage 2 — Word-level matching
        word_scores = self._word_score(tokens)
        total = len(tokens)

        # Stage 3 — Trigram fallback
        tri_scores = self._trigram_score(text)

        # Combine scores: word matches weighted ×2, trigrams weighted ×1
        combined: dict[str, float] = defaultdict(float)
        for lang, score in word_scores.items():
            combined[lang] += score
        for lang, score in tri_scores.items():
            combined[lang] += score * 0.5

        if not combined:
            self.is_reliable = False
            return "en" if total >= 2 else None

        best_lang = max(combined, key=lambda k: combined[k])
        word_hits = word_scores.get(best_lang, 0)
        best_tri = tri_scores.get(best_lang, 0)

        # Confidence: based on word match strength and trigram support
        if word_hits >= 4:
            confidence = 0.90
        elif word_hits >= 2:
            confidence = 0.65
        elif word_hits >= 1:
            confidence = 0.45
        else:
            # Trigram-only detection: need stronger trigram signal
            confidence = min(0.55, best_tri * 0.20)

        # Confidence boost if word + trigram agree
        if word_hits >= 1 and best_tri > 0.3:
            confidence = min(0.95, confidence + 0.12)

        self.is_reliable = confidence >= self._confidence_threshold

        if not self.is_reliable:
            return "en"  # safe default for ambiguous short utterances

        return best_lang

    def detect_with_confidence(self, text: str) -> DetectionResult:
        """Like :meth:`detect` but returns a full :class:`DetectionResult`."""
        code = self.detect(text)
        if code:
            return DetectionResult(
                language=code,
                confidence=0.85 if code != "en" else 0.70,
                is_reliable=self.is_reliable,
            )
        return DetectionResult(
            language="en",
            confidence=0.30,
            is_reliable=False,
        )

    @staticmethod
    def language_name(code: str) -> str:
        """Map an ISO 639-1 code to a human-readable name ('bn' → 'Bangla')."""
        return _LANGUAGE_NAMES.get(code, code)


__all__ = ["LanguageDetector", "DetectionResult", "language_name"]


def language_name(code: str) -> str:
    """Map an ISO 639-1 code to a human-readable name ('bn' → 'Bangla')."""
    return _LANGUAGE_NAMES.get(code, code)
