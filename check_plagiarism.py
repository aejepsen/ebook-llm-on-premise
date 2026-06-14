#!/usr/bin/env python3
"""
Plagiarism / similarity checker — 4-gram overlap analysis.
Compares book chapters (.md) against source PDFs and Jupyter notebooks.
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BOOK_DIR = Path("/home/aejepsen/Documentos/projeto-portifolio/ebook-llm-on-premise/livro")
PDF_DIR = Path("/home/aejepsen/Documentos/ebook")
NB_DIRS = [
    Path("/home/aejepsen/Documentos/git-repo/RAG-with-Python-Cookbook"),
    Path("/home/aejepsen/Documentos/git-repo/llm-model-inference"),
]
REPORT_PATH = Path("/home/aejepsen/Documentos/projeto-portifolio/ebook-llm-on-premise/relatorio_similaridade.md")

NGRAM_SIZE = 4
SUSPECT_THRESHOLD = 0.40  # 40 %
TOP_PARAGRAPHS_PER_CHAPTER = 5
MIN_PARAGRAPH_WORDS = 50

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ngrams(text: str, n: int = NGRAM_SIZE) -> list[tuple[str, ...]]:
    words = normalize(text).split()
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_overlap(book_text: str, source_ngrams_set: set) -> float:
    """Return fraction of book 4-grams that appear in the source set."""
    bg = ngrams(book_text)
    if not bg:
        return 0.0
    matches = sum(1 for g in bg if g in source_ngrams_set)
    return matches / len(bg)


# ---------------------------------------------------------------------------
# Paragraph extraction from Markdown (skip code blocks)
# ---------------------------------------------------------------------------

def extract_paragraphs(md_text: str, min_words: int = MIN_PARAGRAPH_WORDS) -> list[str]:
    """Extract non-code paragraphs with >= min_words words."""
    # Remove fenced code blocks
    cleaned = re.sub(r"```[\s\S]*?```", "", md_text)
    # Remove inline code
    cleaned = re.sub(r"`[^`]+`", "", cleaned)
    # Remove markdown headers markers but keep text
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # Remove images / links markup
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", cleaned)

    paragraphs = []
    for block in re.split(r"\n{2,}", cleaned):
        block = block.strip()
        if not block:
            continue
        words = block.split()
        if len(words) >= min_words:
            paragraphs.append(block)
    return paragraphs


def select_representative(paragraphs: list[str], top_n: int = TOP_PARAGRAPHS_PER_CHAPTER) -> list[str]:
    """Pick the most technical/descriptive paragraphs (longest ones)."""
    ranked = sorted(paragraphs, key=lambda p: len(p.split()), reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Source readers
# ---------------------------------------------------------------------------

def read_pdfs(pdf_dir: Path) -> dict[str, str]:
    """Return {filename: full_text} for each PDF."""
    import pdfplumber

    texts = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
            texts[pdf_path.name] = "\n".join(pages)
            print(f"  [PDF OK] {pdf_path.name}  ({len(pages)} pages)")
        except Exception as e:
            texts[pdf_path.name] = ""
            print(f"  [PDF FAIL] {pdf_path.name}: {e}")
    return texts


def read_notebooks(nb_dirs: list[Path]) -> dict[str, str]:
    """Return {relative_path: markdown_text} for each notebook."""
    texts = {}
    for base in nb_dirs:
        for nb_path in sorted(base.rglob("*.ipynb")):
            try:
                with open(nb_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                cells = nb.get("cells", [])
                md_parts = []
                for cell in cells:
                    if cell.get("cell_type") == "markdown":
                        md_parts.append("".join(cell.get("source", [])))
                rel = str(nb_path.relative_to(base))
                texts[rel] = "\n".join(md_parts)
            except Exception as e:
                rel = str(nb_path.relative_to(base))
                texts[rel] = ""
                print(f"  [NB FAIL] {rel}: {e}")
    print(f"  [NB OK] {len(texts)} notebooks loaded")
    return texts


# ---------------------------------------------------------------------------
# Build combined source n-gram index (per source for attribution)
# ---------------------------------------------------------------------------

def build_source_ngrams(source_texts: dict[str, str]) -> tuple[set, dict[str, set]]:
    """Return (combined_set, {source_name: ngram_set})."""
    per_source: dict[str, set] = {}
    combined: set = set()
    for name, text in source_texts.items():
        if not text:
            continue
        gs = set(ngrams(text))
        per_source[name] = gs
        combined |= gs
    return combined, per_source


def find_best_source(paragraph: str, per_source: dict[str, set]) -> tuple[str, float]:
    """Find which source has highest overlap with this paragraph."""
    bg = ngrams(paragraph)
    if not bg:
        return ("N/A", 0.0)
    best_name, best_score = "N/A", 0.0
    for name, src_set in per_source.items():
        matches = sum(1 for g in bg if g in src_set)
        score = matches / len(bg)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Plagiarism / Similarity Checker  —  4-gram overlap")
    print("=" * 60)

    # 1. Load sources
    print("\n▶ Loading PDFs...")
    pdf_texts = read_pdfs(PDF_DIR)

    print("\n▶ Loading notebooks...")
    nb_texts = read_notebooks(NB_DIRS)

    all_sources = {**{f"[PDF] {k}": v for k, v in pdf_texts.items()},
                   **{f"[NB] {k}": v for k, v in nb_texts.items()}}

    print("\n▶ Building source n-gram index...")
    combined_set, per_source = build_source_ngrams(all_sources)
    print(f"  Combined n-gram set size: {len(combined_set):,}")

    # 2. Analyse each chapter
    print("\n▶ Analysing chapters...\n")

    chapter_files = sorted(BOOK_DIR.glob("cap*.md"))
    results: list[dict] = []
    total_paragraphs = 0
    total_suspect = 0

    for cap_path in chapter_files:
        cap_name = cap_path.stem
        md_text = cap_path.read_text(encoding="utf-8")
        all_paras = extract_paragraphs(md_text)
        representative = select_representative(all_paras)

        chapter_result = {
            "chapter": cap_name,
            "total_paragraphs": len(all_paras),
            "analysed": len(representative),
            "suspects": [],
            "scores": [],
        }

        for para in representative:
            score = ngram_overlap(para, combined_set)
            chapter_result["scores"].append(score)
            best_src, best_score = find_best_source(para, per_source)
            if score >= SUSPECT_THRESHOLD:
                chapter_result["suspects"].append({
                    "score": score,
                    "best_source": best_src,
                    "best_source_score": best_score,
                    "excerpt": para[:300],
                })
                total_suspect += 1

        total_paragraphs += len(representative)
        avg = sum(chapter_result["scores"]) / len(chapter_result["scores"]) if chapter_result["scores"] else 0
        chapter_result["avg_score"] = avg
        chapter_result["originality"] = 1.0 - avg
        results.append(chapter_result)

        flag = " ⚠" if chapter_result["suspects"] else " ✓"
        print(f"  {cap_name:40s}  avg={avg:.1%}  suspects={len(chapter_result['suspects'])}{flag}")

    # 3. Write report
    print(f"\n▶ Writing report to {REPORT_PATH}")

    lines = [
        f"# Relatório de Similaridade — 4-gram Overlap",
        f"",
        f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"",
        f"## Resumo",
        f"",
        f"| Métrica | Valor |",
        f"|---|---|",
        f"| Capítulos analisados | {len(results)} |",
        f"| Parágrafos analisados (top-5/cap) | {total_paragraphs} |",
        f"| Parágrafos suspeitos (>40% match) | {total_suspect} |",
        f"| Fontes PDF | {len(pdf_texts)} |",
        f"| Fontes Notebook | {len(nb_texts)} |",
        f"| N-grams únicos nas fontes | {len(combined_set):,} |",
        f"",
        f"## Originalidade por Capítulo",
        f"",
        f"| Capítulo | Parágrafos | Avg 4-gram match | Originalidade | Suspeitos |",
        f"|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['chapter']} | {r['total_paragraphs']} | {r['avg_score']:.1%} | {r['originality']:.1%} | {len(r['suspects'])} |"
        )

    lines.append("")
    lines.append("## Parágrafos Suspeitos (>40% 4-gram overlap)")
    lines.append("")

    if total_suspect == 0:
        lines.append("Nenhum parágrafo suspeito encontrado.")
    else:
        for r in results:
            if not r["suspects"]:
                continue
            lines.append(f"### {r['chapter']}")
            lines.append("")
            for i, s in enumerate(r["suspects"], 1):
                lines.append(f"**Suspeito {i}** — match: {s['score']:.1%} — fonte mais similar: `{s['best_source']}` ({s['best_source_score']:.1%})")
                lines.append("")
                lines.append(f"> {s['excerpt']}...")
                lines.append("")

    lines.append("## Notas Metodológicas")
    lines.append("")
    lines.append("- **N-gram size**: 4 palavras")
    lines.append("- **Threshold de suspeita**: 40% de 4-grams do parágrafo presentes nas fontes")
    lines.append("- **Seleção de parágrafos**: top-5 mais longos por capítulo (>50 palavras, sem código)")
    lines.append("- **Normalização**: lowercase, remoção de pontuação, colapso de espaços")
    lines.append("- Overlap alto em parágrafos técnicos pode ser natural (terminologia compartilhada)")
    lines.append("- PDFs que falharam na leitura foram ignorados na comparação")
    lines.append("")

    # Check for failed PDFs
    failed_pdfs = [k for k, v in pdf_texts.items() if not v]
    if failed_pdfs:
        lines.append("### PDFs que falharam na leitura")
        lines.append("")
        for fp in failed_pdfs:
            lines.append(f"- {fp}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Done. Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
