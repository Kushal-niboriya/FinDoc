"""
Preprocessing module for FinDoc: Legal & Financial Clause Intelligence.
Supports:
1. Traditional NLP pipeline: lowercasing, tokenization, stopword removal, lemmatization (all toggleable).
2. Specialized Legal NLP pipeline:
   - Clause numbering and section reference removal
   - Typed entity masking: [MONEY], [DATE], [DURATION]
   - Defined terms and party reference normalization: [PARTY]
   - SEC formatting artifact and whitespace collapse
   - Deduplication handling
"""

import re
from typing import List, Optional, Set, Callable, Tuple
import spacy

# Compile regex patterns once for maximum performance

# 1. Clause Numbering and Section Cross-References
# Matches: "Section 12.4(a)", "Section 1.1", "(iv)", "1.01.", "Article IV."
RE_SECTION_REF = re.compile(
    r"\b(?:Section|Article|Clause|Paragraph|Schedule|Exhibit)\s+\d+(?:[\.\-]\d+)*(?:\([a-zA-Z0-9]+\))*",
    re.IGNORECASE
)
RE_CLAUSE_NUMBERING = re.compile(
    r"^(?:\s*(?:\d+[\.\)]|\([a-zA-Z0-9]+\)|[ivxlcdmIVXLCDM]+[\.\)]))+\s*",
    re.MULTILINE
)
RE_INLINE_NUMBERING = re.compile(
    r"\s+\([a-zA-Z0-9ivxlcdm]+\)\s+"
)

# 2. Money / Currencies
# Matches: $10,000,000, $5.5 million, USD 500,000, $ 100
RE_MONEY = re.compile(
    r"(?:\$|USD|EUR|GBP|¥)\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand))?",
    re.IGNORECASE
)

# 3. Dates
# Matches: "January 15, 2024", "15th day of March, 2023", "12/31/2022"
RE_DATE_FORMAL = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
    re.IGNORECASE
)
RE_DATE_DAY_OF = re.compile(
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4}\b",
    re.IGNORECASE
)
RE_DATE_NUMERIC = re.compile(
    r"\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b"
)

# 4. Durations
# Matches: "thirty (30) days", "180 calendar days", "3 consecutive years", "10 business days"
RE_DURATION_WORDS_DIGITS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|sixty|ninety|one hundred)\s*\(\d+\)\s*(?:business\s+|calendar\s+)?(?:days|months|years|weeks|hours)\b",
    re.IGNORECASE
)
RE_DURATION_NUMERIC = re.compile(
    r"\b\d+\s*(?:business\s+|calendar\s+)?(?:days|months|years|weeks|hours)\b",
    re.IGNORECASE
)

# 5. Defined Terms and Party References
# Matches: "the Company", "the Borrower", "the Lender", "the Executive", "the Purchaser", "the Seller"
RE_PARTY_DEFINED = re.compile(
    r"\bthe\s+(?:Company|Borrower|Lender|Executive|Purchaser|Seller|Guarantor|Employer|Employee|Optionee|Indemnitee|Tenant|Landlord|Assignor|Assignee)\b"
)
RE_QUOTED_DEFINED = re.compile(
    r"[\"“\'](?:Company|Borrower|Lender|Executive|Purchaser|Seller|Guarantor|Employer|Employee)[\"”\']"
)

# 6. SEC Boilerplate Artifacts and Whitespace
RE_BOILERPLATE_CHARS = re.compile(r"[_=\-\*]{3,}")
RE_WHITESPACE = re.compile(r"\s+")
RE_WORD_TOKENS = re.compile(r"\b\w+\b")


def tokenize_simple(text: str) -> List[str]:
    """Fast word-boundary tokenization."""
    return RE_WORD_TOKENS.findall(text.lower())

# Lazy-loaded spaCy model for lemmatization
_SPACY_NLP = None


def get_spacy_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _SPACY_NLP


class PreprocessingPipeline:
    """
    Configurable, modular preprocessor for legal contract clauses.
    """

    def __init__(
        self,
        # Traditional settings
        lowercase: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        # Specialized legal settings
        remove_numbering: bool = False,
        mask_money: bool = False,
        mask_dates: bool = False,
        mask_durations: bool = False,
        normalize_parties: bool = False,
        collapse_boilerplate: bool = False,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        self.remove_numbering = remove_numbering
        self.mask_money = mask_money
        self.mask_dates = mask_dates
        self.mask_durations = mask_durations
        self.normalize_parties = normalize_parties
        self.collapse_boilerplate = collapse_boilerplate

    def transform_text(self, text: str) -> str:
        """Applies configured pipeline steps to a single raw clause text."""
        if not text:
            return ""

        # Step 1: Specialized boilerplate cleanup
        if self.collapse_boilerplate:
            text = RE_BOILERPLATE_CHARS.sub(" ", text)

        # Step 2: Specialized numbering and section removal
        if self.remove_numbering:
            text = RE_CLAUSE_NUMBERING.sub("", text)
            text = RE_SECTION_REF.sub("[REF]", text)
            text = RE_INLINE_NUMBERING.sub(" ", text)

        # Step 3: Typed entity masking
        if self.mask_money:
            text = RE_MONEY.sub("[MONEY]", text)

        if self.mask_dates:
            text = RE_DATE_DAY_OF.sub("[DATE]", text)
            text = RE_DATE_FORMAL.sub("[DATE]", text)
            text = RE_DATE_NUMERIC.sub("[DATE]", text)

        if self.mask_durations:
            text = RE_DURATION_WORDS_DIGITS.sub("[DURATION]", text)
            text = RE_DURATION_NUMERIC.sub("[DURATION]", text)

        # Step 4: Party normalization
        if self.normalize_parties:
            text = RE_QUOTED_DEFINED.sub("[PARTY]", text)
            text = RE_PARTY_DEFINED.sub("the [PARTY]", text)

        # Step 5: Whitespace normalization
        text = RE_WHITESPACE.sub(" ", text).strip()

        # Step 6: Lowercasing (preserves uppercase tokens like [MONEY], [DATE], [PARTY] if needed or lowers all)
        if self.lowercase:
            text = text.lower()

        # Step 7: Lemmatization / Stopwords via spaCy if enabled
        if self.lemmatize or self.remove_stopwords:
            nlp = get_spacy_nlp()
            doc = nlp(text)
            tokens = []
            for token in doc:
                if self.remove_stopwords and token.is_stop:
                    continue
                if token.is_punct and self.collapse_boilerplate:
                    continue
                t = token.lemma_ if self.lemmatize else token.text
                if t.strip():
                    tokens.append(t)
            text = " ".join(tokens)

        return text

    def transform_batch(self, texts: List[str], batch_size: int = 1000) -> List[str]:
        """Processes a list of texts efficiently."""
        if not (self.lemmatize or self.remove_stopwords):
            # Fast pure-python regex path
            return [self.transform_text(t) for t in texts]

        # Use spaCy batch pipeline if lemmatization is requested
        nlp = get_spacy_nlp()
        # First apply regex pre-transformations
        pre_processed = []
        for t in texts:
            # Run non-spacy steps
            if self.collapse_boilerplate:
                t = RE_BOILERPLATE_CHARS.sub(" ", t)
            if self.remove_numbering:
                t = RE_CLAUSE_NUMBERING.sub("", t)
                t = RE_SECTION_REF.sub("[ref]", t)
                t = RE_INLINE_NUMBERING.sub(" ", t)
            if self.mask_money:
                t = RE_MONEY.sub("[money]", t)
            if self.mask_dates:
                t = RE_DATE_DAY_OF.sub("[date]", t)
                t = RE_DATE_FORMAL.sub("[date]", t)
                t = RE_DATE_NUMERIC.sub("[date]", t)
            if self.mask_durations:
                t = RE_DURATION_WORDS_DIGITS.sub("[duration]", t)
                t = RE_DURATION_NUMERIC.sub("[duration]", t)
            if self.normalize_parties:
                t = RE_QUOTED_DEFINED.sub("[party]", t)
                t = RE_PARTY_DEFINED.sub("the [party]", t)
            t = RE_WHITESPACE.sub(" ", t).strip()
            if self.lowercase:
                t = t.lower()
            pre_processed.append(t)

        results = []
        for doc in nlp.pipe(pre_processed, batch_size=batch_size):
            tokens = []
            for token in doc:
                if self.remove_stopwords and token.is_stop:
                    continue
                t = token.lemma_ if self.lemmatize else token.text
                if t.strip():
                    tokens.append(t)
            results.append(" ".join(tokens))
        return results


def deduplicate_corpus(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int], int]:
    """
    Deduplication policy: keeps first occurrence of identical normalized text.
    Returns (deduped_texts, deduped_labels, num_removed).
    """
    seen = set()
    out_texts = []
    out_labels = []
    removed = 0

    for t, l in zip(texts, labels):
        norm = re.sub(r"\s+", " ", t.strip().lower())
        if norm not in seen:
            seen.add(norm)
            out_texts.append(t)
            out_labels.append(l)
        else:
            removed += 1

    return out_texts, out_labels, removed
