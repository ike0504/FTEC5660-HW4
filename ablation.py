"""Reproduce the Task 5 ablation without changing the provided hw4.py runner.

Retrieval-only evaluation works without an API key. Add --generate to call the fixed
DeepSeek generator and measure end-to-end answer accuracy as well.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


def recursive_chunks(file_name: str, html: str, structured_tables: bool) -> list[Document]:
    soup = BeautifulSoup(html, "lxml")
    company, year, doc_type = hw4._document_metadata(soup)
    if structured_tables:
        for table in soup.find_all("table"):
            table.replace_with(NavigableString("\n" + hw4.table_to_text(str(table)) + "\n"))
    text = soup.get_text(" ", strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return [Document(page_content=part, metadata={"source": file_name, "company": company,
                                                   "fiscal_year": year, "doc_type": doc_type})
            for part in splitter.split_text(text)]


def make_index(chunks: list[Document], canonical_tokens: bool) -> dict:
    tokenize = hw4._tokenize if canonical_tokens else lambda text: text.lower().split()
    tokens = [tokenize(c.page_content) for c in chunks]
    index = {"chunks": chunks, "tokens": tokens, "bm25": BM25Okapi(tokens),
             "token_sets": [set(t) for t in tokens], "by_company": {}}
    for i, chunk in enumerate(chunks):
        key = hw4._company_key_local(chunk.metadata["company"])
        index["by_company"].setdefault(key, []).append(i)
    return index


def global_top5(index: dict, question: str, canonical_tokens: bool = False) -> list[int]:
    tokens = hw4._tokenize(question) if canonical_tokens else question.lower().split()
    scores = index["bm25"].get_scores(tokens)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]


def filtered_ids(index: dict, parsed: dict) -> list[int]:
    ids: list[int] = []
    for company in parsed["companies"]:
        key = hw4._company_key_local(company)
        ids.extend(hw4._candidate_ids(index, key, parsed))
    return list(dict.fromkeys(ids))


def lexical_rank(index: dict, ids: list[int], question: str, canonical: bool, enhanced: bool,
                 year: int | None) -> list[int]:
    if enhanced:
        return hw4._rank_candidates(index, ids, question, year)
    tokens = hw4._tokenize(question) if canonical else question.lower().split()
    scores = index["bm25"].get_scores(tokens)
    return sorted(ids, key=lambda i: scores[i], reverse=True)


def retrieve_variant(index: dict, question: str, variant: str) -> list[Document]:
    if variant in {"B0", "V1", "V2"}:
        ids = global_top5(index, question)
    else:
        parsed = hw4.parse_query(question)
        canonical = variant in {"V4", "V5", "V6"}
        enhanced = variant in {"V5", "V6"}
        ids = filtered_ids(index, parsed)
        # Ablations B0-V6 never abstain. Fall back when metadata has no match.
        if not ids:
            ids = global_top5(index, question)
        elif variant == "V6" and len(parsed["companies"]) > 1:
            rankings = []
            for company in parsed["companies"]:
                company_ids = hw4._candidate_ids(index, hw4._company_key_local(company), parsed)
                rankings.append(hw4._rank_candidates(index, company_ids, question, parsed["fiscal_year"]))
            ids = []
            for depth in range(2):
                for ranking in rankings:
                    if depth < len(ranking):
                        ids.append(ranking[depth])
            remaining = [i for ranking in rankings for i in ranking[2:] if i not in ids]
            if remaining:
                ids.append(remaining[0])
            ids = ids[:5]
        else:
            ids = lexical_rank(index, ids, question, canonical, enhanced, parsed["fiscal_year"])[:5]
    return [index["chunks"][i] for i in ids[:5]]


def evaluate(variant: str, corpus: dict[str, str], questions: dict[str, str], truth: dict,
             docs_meta: dict, generate: bool) -> tuple[dict, dict[str, list[Document]], dict[str, str]]:
    if variant == "B0":
        by_file = {f: recursive_chunks(f, html, False) for f, html in corpus.items()}
    elif variant == "V1":
        by_file = {f: recursive_chunks(f, html, True) for f, html in corpus.items()}
    else:
        by_file = {f: hw4.chunk_document(f, html) for f, html in corpus.items()}
    chunks = [c for f in sorted(by_file) for c in by_file[f]]
    doc_pass = sum(not hw4.document_problems(by_file[f], None, docs_meta[f], f) for f in docs_meta) / len(docs_meta)
    if variant == "V7":
        index = hw4.build_index(chunks)
        retrieved = {qid: hw4.retrieve(index, q) for qid, q in questions.items()}
    else:
        index = make_index(chunks, variant in {"V4", "V5", "V6"})
        retrieved = {qid: retrieve_variant(index, q, variant) for qid, q in questions.items()}
    if generate:
        def answer(qid: str) -> str:
            return hw4.generate(questions[qid], retrieved[qid])
        with ThreadPoolExecutor(max_workers=8) as pool:
            answers = dict(zip(questions, pool.map(answer, questions)))
    else:
        answers = {qid: "" for qid in questions}
    answerable = [qid for qid, gold in truth.items() if gold["evidence"]]
    unanswerable = [qid for qid, gold in truth.items() if not gold["evidence"]]
    recall = sum(hw4.retrieval_hit(retrieved[qid], truth[qid]) for qid in answerable) / len(answerable)
    kept = sum(bool(retrieved[qid]) for qid in answerable) / len(answerable)
    rejected = sum(not retrieved[qid] for qid in unanswerable) / len(unanswerable)
    parse_mean = sum(hw4.parse_score(hw4.parse_query(questions[qid]), truth[qid]) for qid in truth) / len(truth)
    result = {
        "variant": variant,
        "documents_passing": f"{doc_pass:.1%}",
        "parse_fields": f"{parse_mean:.1%}" if variant >= "V3" else "-",
        "recall_at_5": f"{recall:.1%}",
        "answerable_kept": f"{kept:.1%}",
        "unanswerable_rejected": f"{rejected:.1%}",
        "abstention_score": f"{max(0.0, kept + rejected - 1):.1%}",
        "answer_accuracy": f"{sum(hw4.answer_correct(answers[qid], truth[qid]) for qid in truth) / len(truth):.1%}" if generate else "not run",
    }
    return result, retrieved, answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--generate", action="store_true", help="also run the fixed DeepSeek generator")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()
    if args.generate and not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY before using --generate.")
    folder = Path(args.data_folder)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    truth = hw4.load_json(folder / "ground_truth.json")
    docs_meta = hw4.load_json(folder / "docs.json")
    rows = []
    final_retrieved = final_answers = None
    for variant in ["B0", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]:
        row, retrieved, answers = evaluate(variant, corpus, questions, truth, docs_meta, args.generate)
        rows.append(row)
        print(row)
        if variant == "V7":
            final_retrieved, final_answers = retrieved, answers
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    detail_name = "results.csv" if args.generate else "retrieval_results.csv"
    detail_path = Path(args.output).with_name(detail_name)
    with open(detail_path, "w", newline="", encoding="utf-8") as handle:
        fields = ["id", "category", "question", "retrieved", "retrieval_hit", "answer", "correct", "expected"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid, question in questions.items():
            gold = truth[qid]
            hit = hw4.retrieval_hit(final_retrieved[qid], gold) if gold["evidence"] else ""
            answer = final_answers[qid]
            correct = hw4.answer_correct(answer, gold) if args.generate else ""
            writer.writerow({"id": qid, "category": gold["category"], "question": question,
                             "retrieved": ";".join(c.metadata["source"] for c in final_retrieved[qid]),
                             "retrieval_hit": hit, "answer": answer, "correct": correct,
                             "expected": "" if correct is True else gold["answer"]})
    print(f"Wrote {args.output} and {detail_path}")


if __name__ == "__main__":
    main()
