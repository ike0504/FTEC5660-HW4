"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================
from collections import defaultdict

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi


# chunk_document populates this registry before any query is parsed.  Keys contain no
# punctuation or legal suffix; values are the names as printed in the documents.
_COMPANY_REGISTRY: dict[str, str] = {}
_LEGAL_SUFFIX_RE = re.compile(
    r"\s+(?:inc|corp|corporation|ltd|limited|plc|llc|co)\.?$", re.IGNORECASE
)


def _clean_text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _company_key_local(name: Any) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", str(name).casefold()).split()
    while words and words[-1] in {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}:
        words.pop()
    return " ".join(words)


def _display_company(name: str) -> str:
    return _LEGAL_SUFFIX_RE.sub("", name).strip()


def _document_metadata(soup: BeautifulSoup) -> tuple[str, int, str]:
    """Read the cover/title metadata shared by all three document layouts."""
    h1 = soup.find("h1")
    company = _clean_text(h1) if h1 else ""
    company = re.sub(r"\s*\([^)]*\)\s*$", "", company).strip()
    subtitle = soup.find(class_="subtitle")
    subtitle_text = _clean_text(subtitle) if subtitle else ""
    h2_text = _clean_text(soup.find("h2")) if soup.find("h2") else ""
    title_text = _clean_text(soup.find("title")) if soup.find("title") else ""

    if "annual report" in subtitle_text.casefold():
        doc_type, year_text = "annual_report", subtitle_text
    elif re.search(r"earnings (?:conference )?call", h2_text + " " + title_text, re.IGNORECASE):
        doc_type, year_text = "earnings_call", h2_text or title_text
    else:
        doc_type, year_text = "research_report", h2_text or title_text
    year_match = re.search(r"(?<!\d)(20\d{2})\b", year_text)
    if not year_match:
        year_match = re.search(r"(?<!\d)(20\d{2})\b", title_text)
    fiscal_year = int(year_match.group(1)) if year_match else 0
    return company, fiscal_year, doc_type


def _split_long_text(text: str, limit: int) -> list[str]:
    """Split at a word/phrase boundary, without discarding or inventing text."""
    pieces: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        cut = max(rest.rfind("\n", 0, limit + 1), rest.rfind(". ", 0, limit + 1),
                  rest.rfind(" ", 0, limit + 1))
        if cut < limit // 2:
            cut = limit
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table")
    if table is None:
        return ""
    caption_node = table.find("caption")
    caption = _clean_text(caption_node) if caption_node else ""
    header_row = table.find("thead").find("tr") if table.find("thead") else None
    if header_row is None:
        first_row = table.find("tr")
        if first_row and first_row.find("th"):
            header_row = first_row
    headers = [_clean_text(c) for c in header_row.find_all(["th", "td"], recursive=False)] if header_row else []

    lines = [caption] if caption else []
    rows = table.find_all("tr")
    for row in rows:
        if header_row is not None and (row is header_row or row.find_parent("thead")):
            continue
        cells = row.find_all(["th", "td"], recursive=False)
        values = [_clean_text(c) for c in cells]
        if not values:
            continue
        if len(values) == 1:  # a section row such as "Current assets"
            lines.append(values[0])
            continue
        label, row_values = values[0], values[1:]
        if len(headers) >= len(values):
            value_headers = headers[1:1 + len(row_values)]
        elif len(headers) == len(row_values):
            value_headers = headers
        else:
            value_headers = [f"Column {i + 2}" for i in range(len(row_values))]
        fields = [f"{head}: {value}" if head else value
                  for head, value in zip(value_headers, row_values)]
        # Very wide rows are wrapped only when necessary; every wrapped line repeats its row label.
        current = label
        for field in fields:
            candidate = current + " | " + field
            if len(candidate) > 390 and current != label:
                lines.append(current)
                current = label + " | " + field
            else:
                current = candidate
        lines.append(current)
    return "\n".join(line for line in lines if line)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "lxml")
    company, fiscal_year, doc_type = _document_metadata(soup)
    _COMPANY_REGISTRY[_company_key_local(company)] = _display_company(company)
    metadata = {"source": file_name, "company": company,
                "fiscal_year": fiscal_year, "doc_type": doc_type}
    for unwanted in soup(["nav", "style", "script"]):
        unwanted.decompose()
    root = soup.body or soup
    chunks: list[Document] = []
    buffer = ""
    buffer_section = ""

    def emit(text: str, section: str, extra: dict | None = None) -> None:
        if not text.strip():
            return
        context = f"{company} | FY{fiscal_year} | {doc_type.replace('_', ' ')}"
        if section:
            context += f" | {section}"
        prefix = context + "\n"
        meta = {**metadata, "section": section, **(extra or {})}
        for piece in _split_long_text(text, MAX_CHUNK_CHARS - len(prefix)):
            chunks.append(Document(page_content=prefix + piece, metadata=dict(meta)))

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            emit(buffer.strip(), buffer_section)
        buffer = ""

    current_h2 = current_h3 = ""
    tags = root.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"])
    for tag in tags:
        if tag.find_parent(["table", "ul", "ol"]):
            continue
        section = " > ".join(x for x in (current_h2, current_h3) if x)
        if tag.name == "h2":
            flush()
            current_h2, current_h3 = _clean_text(tag), ""
            continue
        if tag.name == "h3":
            flush()
            current_h3 = _clean_text(tag)
            continue
        if tag.name == "table":
            flush()
            table_text = table_to_text(str(tag))
            table_lines = table_text.splitlines()
            caption = table_lines[0] if table_lines else ""
            body_lines = table_lines[1:] if caption else table_lines
            table_class = (tag.get("class") or [""])[0]
            context = f"{company} | FY{fiscal_year} | {doc_type.replace('_', ' ')}"
            if section:
                context += f" | {section}"
            prefix = context + "\n" + caption + ("\n" if caption else "")
            room = MAX_CHUNK_CHARS - len(prefix)
            group: list[str] = []
            size = 0
            for line in body_lines:
                if group and size + len(line) + 1 > room:
                    emit(prefix.split("\n", 1)[1] + "\n".join(group), section, {"table_type": table_class})
                    group, size = [], 0
                group.append(line)
                size += len(line) + 1
            if group or not body_lines:
                emit(prefix.split("\n", 1)[1] + "\n".join(group), section, {"table_type": table_class})
            continue
        text = _clean_text(tag)
        if not text:
            continue
        if buffer and (section != buffer_section or len(buffer) + len(text) + 2 > 1150):
            flush()
        buffer_section = section
        buffer += ("\n" if buffer else "") + text
    flush()
    return chunks


def _extract_company_mentions(question: str) -> list[str]:
    """Match known entities first, then retain capitalised unknown company candidates."""
    found: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for key, display in sorted(_COMPANY_REGISTRY.items(), key=lambda item: -len(item[0])):
        pattern = re.escape(display).replace(r"\ ", r"\s+")
        match = re.search(rf"(?<![A-Za-z0-9]){pattern}(?:{_LEGAL_SUFFIX_RE.pattern})?(?![A-Za-z0-9])",
                          question, re.IGNORECASE)
        if match:
            found.append((match.start(), display))
            occupied.append((match.start(), match.end()))
    # Comparisons and possessives are reliable anchors for companies absent from the registry.
    candidates: list[tuple[int, str]] = []
    comparison = re.search(r":\s*([^?:]+?)\s+or\s+([^?:]+?)\s*\?", question)
    if comparison:
        for match in re.finditer(r"[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)*", comparison.group(0)):
            candidates.append((comparison.start() + match.start(), match.group(0)))
    capitalised = re.compile(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*){1,5}\b")
    candidates.extend((m.start(), m.group(0)) for m in capitalised.finditer(question))
    excluded = {
        "chief executive officer", "chief financial officer", "chief operating officer",
        "chief technology officer", "asia pacific", "middle east", "equity research",
        "fourth quarter", "full year", "annual report", "financial statements",
    }
    excluded_last = {"capital", "securities", "research", "llp", "officer", "statements"}
    for pos, raw in sorted(candidates):
        raw = raw.strip(" ,.;:")
        clean = _LEGAL_SUFFIX_RE.sub("", raw).strip()
        key = _company_key_local(clean)
        if not key or key in {k for k, _ in found} or key.casefold() in excluded:
            continue
        if key.split()[-1] in excluded_last or any(a <= pos < b for a, b in occupied):
            continue
        # Generic capitalised phrases are not companies; a company mention usually has an anchor.
        before = question[max(0, pos - 35):pos].casefold()
        after = question[pos + len(raw):pos + len(raw) + 25].casefold()
        anchored = bool(re.search(r"(?:for|of|did|does|give|audited|:|or)\s*$", before)) or after.startswith(("'", " or", " hold"))
        if anchored or key in _COMPANY_REGISTRY:
            found.append((pos, clean))
            occupied.append((pos, pos + len(raw)))
    unique: dict[str, tuple[int, str]] = {}
    for pos, name in found:
        unique.setdefault(_company_key_local(name), (pos, name))
    return [name for _, name in sorted(unique.values())]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    year_match = re.search(r"\b(?:FY\s*)?(20\d{2})\b", question, re.IGNORECASE)
    fiscal_year = int(year_match.group(1)) if year_match else None
    lowered = question.casefold()
    if re.search(r"earnings (?:conference )?call|\bq4\b.*\bcall\b", lowered):
        doc_type = "earnings_call"
    elif re.search(r"results review|target price|\binvestment rating\b|\bbroker\b|\bforecast\b|\bestimate\b", lowered):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    return {"companies": _extract_company_mentions(question),
            "fiscal_year": fiscal_year, "doc_type": doc_type}


_TOKEN_RE = re.compile(r"[a-z]+|\d[\d,]*(?:\.\d+)?%?", re.IGNORECASE)
_TOKEN_CANONICAL = {
    "audited": "audit", "auditor": "audit", "auditors": "audit", "auditing": "audit",
    "headquartered": "headquarter", "headquarters": "headquarter",
    "borrowings": "borrow", "borrowing": "borrow", "inventories": "inventory",
    "employees": "employee", "employed": "employee", "receivables": "receivable",
    "repurchases": "repurchase", "program": "programme", "estimates": "estimate",
}


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold()):
        if raw[0].isdigit():
            tokens.append(raw)
            if "," in raw:
                tokens.append(raw.replace(",", ""))
            continue
        token = _TOKEN_CANONICAL.get(raw, raw)
        if token not in _TOKEN_CANONICAL.values() and len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif token not in _TOKEN_CANONICAL.values() and len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


_EXPANSION_RULES = [
    (r"net accounts receivable|accounts receivable, net", "accounts receivable net"),
    (r"operating cash flow|net cash from operating activities", "net cash from operating activities"),
    (r"\bemployees\b|\bheadcount\b", "total employees number of people"),
    (r"long[- ]term borrowings?", "long term borrowings"),
    (r"share repurchase|buyback", "share repurchase programme buyback"),
    (r"headquartered|headquarters", "headquarters"),
    (r"audit firm|auditor", "independent auditor audited financial statements"),
    (r"\bforecast\b", "estimate estimates"),
    (r"target price", "12 month target price"),
    (r"investment rating", "investment rating"),
]

# Question wording that does not itself have to occur in evidence.  Keeping this list
# domain-general is important: the private corpus contains different companies and values.
_QUERY_STOPWORDS = set("""
a about an and are as at be been broker by call company companies did do does earn earned end
financial fiscal for forecast from generate generated give had has have held higher his hold how in inc
investment is it its large long million millions much more most of on or programme q4 report review
set share shares should than that the their to was were what when which who whom why with year
usd value amount announce announced management guide guided spend spent end fy results review fourth
quarter full earnings conference research equity 12-month city firm shares
""".split())


def _expanded_query(question: str) -> str:
    additions = [extra for pattern, extra in _EXPANSION_RULES
                 if re.search(pattern, question, re.IGNORECASE)]
    return question + (" " + " ".join(additions) if additions else "")


def _salient_query_tokens(question: str, companies: list[str]) -> set[str]:
    stripped = question
    for company in companies:
        stripped = re.sub(re.escape(company), " ", stripped, flags=re.IGNORECASE)
    # Remove any remaining capitalised company phrase so its tokens cannot trigger abstention.
    stripped = re.sub(r"\b[A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)+\b", " ", stripped)
    return {t for t in _tokenize(stripped)
            if not t[0].isdigit() and t not in _QUERY_STOPWORDS and len(t) > 1}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    for chunk in chunks:
        company = chunk.metadata.get("company", "")
        _COMPANY_REGISTRY.setdefault(_company_key_local(company), _display_company(company))
    tokenised = [_tokenize(c.page_content) for c in chunks]
    by_company: dict[str, list[int]] = defaultdict(list)
    for i, chunk in enumerate(chunks):
        by_company[_company_key_local(chunk.metadata.get("company", ""))].append(i)
    return {"chunks": chunks, "tokens": tokenised, "bm25": BM25Okapi(tokenised, k1=1.5, b=0.75),
            "by_company": dict(by_company), "token_sets": [set(t) for t in tokenised]}


def _candidate_ids(index: Any, company_key: str, parsed: dict) -> list[int]:
    ids = list(index["by_company"].get(company_key, []))
    ids = [i for i in ids if index["chunks"][i].metadata.get("doc_type") == parsed["doc_type"]]
    year = parsed["fiscal_year"]
    if not ids or year is None or parsed["doc_type"] != "annual_report":
        return ids
    exact = [i for i in ids if index["chunks"][i].metadata.get("fiscal_year") == year]
    if exact:
        return exact
    # A prior-year column is available in the following year's annual report.
    following = [i for i in ids if index["chunks"][i].metadata.get("fiscal_year") == year + 1]
    return following


def _rank_candidates(index: Any, ids: list[int], question: str, fiscal_year: int | None) -> list[int]:
    query_tokens = _tokenize(_expanded_query(question))
    if not ids or not query_tokens:
        return []
    scores = index["bm25"].get_scores(query_tokens)
    query_set = set(query_tokens)
    parsed = parse_query(question)
    salient = _salient_query_tokens(question, parsed["companies"])
    ranked = []
    for i in ids:
        token_set = index["token_sets"][i]
        overlap = len(query_set & token_set) / max(len(query_set), 1)
        concept_overlap = len(salient & token_set) / max(len(salient), 1) if salient else 0.0
        year_boost = 1.5 if fiscal_year is not None and str(fiscal_year) in token_set else 0.0
        # Distinctive query concepts outweigh generic words such as "value" or "company".
        score = float(scores[i]) + 4.0 * overlap + 10.0 * concept_overlap + year_boost
        ranked.append((score, -i, i))
    return [i for _, _, i in sorted(ranked, reverse=True)]


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    parsed = parse_query(question)
    companies = parsed["companies"]
    if not companies:
        return []
    company_keys = [_company_key_local(c) for c in companies]
    # An explicitly named company outside the corpus must not be resolved to a similar name.
    if any(key not in index["by_company"] for key in company_keys):
        return []
    candidates_by_company = [_candidate_ids(index, key, parsed) for key in company_keys]
    if any(not ids for ids in candidates_by_company):
        return []

    salient = _salient_query_tokens(question, companies)
    for ids in candidates_by_company:
        available_tokens = set().union(*(index["token_sets"][i] for i in ids))
        # Missing distinctive terms identify undisclosed concepts without a list of hard-coded
        # unanswerable topics.  Also reject a named year that the selected document type never has.
        if salient - available_tokens:
            return []
        if parsed["fiscal_year"] is not None and str(parsed["fiscal_year"]) not in available_tokens:
            return []

    rankings = [_rank_candidates(index, ids, question, parsed["fiscal_year"])
                for ids in candidates_by_company]
    if any(not ranking for ranking in rankings):
        return []
    selected: list[int] = []
    if len(rankings) == 1:
        selected = rankings[0][:MAX_RETRIEVED]
    else:
        # Reserve two passages for every compared company before filling the fifth slot.
        per_company = max(1, (MAX_RETRIEVED - 1) // len(rankings))
        for depth in range(per_company):
            for ranking in rankings:
                if depth < len(ranking):
                    selected.append(ranking[depth])
        remaining = [i for ranking in rankings for i in ranking[per_company:] if i not in selected]
        if remaining and len(selected) < MAX_RETRIEVED:
            selected.append(remaining[0])
    return [index["chunks"][i] for i in selected[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
