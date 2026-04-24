"""
PDF ingestion with section-aware chunking for semiconductor documentation.

Design decisions (important for interviews):
- pdfplumber over pypdf: better table extraction, critical for datasheets where
  timing specs and electrical characteristics live in tables.
- Section-aware chunking: preserves spec hierarchy (e.g., "Electrical Characteristics
  > DC Specs > VDD"). Blind fixed-size chunking destroys this structure.
- Chunk size 500-800 tokens with 100-token overlap: balances retrieval precision
  (smaller is better for exact spec lookup) with context (larger is better for
  conceptual queries).
- Metadata preserved per chunk: doc_name, section, page_num. Enables filtered
  retrieval and source attribution.
"""

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pdfplumber
import tiktoken
from tqdm import tqdm


@dataclass
class Chunk:
    """A single retrievable chunk with metadata."""
    chunk_id: str
    text: str
    doc_name: str
    section: str
    page_num: int
    token_count: int
    chip_family: Optional[str] = None  # e.g., "cortex-m4", "risc-v", "h100"


# Heading detection regex. Matches common datasheet heading patterns:
# - "1.2.3 Section Title"
# - "Chapter 4: Electrical Characteristics"
# - ALL CAPS headings (common in datasheets)
HEADING_PATTERNS = [
    re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][A-Za-z0-9\s\-,:/()]+)$"),  # "1.2.3 Title"
    re.compile(r"^\s*Chapter\s+\d+[:\s]+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*Section\s+\d+[:\s]+(.+)$", re.IGNORECASE),
    re.compile(r"^\s*([A-Z][A-Z\s\-]{4,})\s*$"),  # ALL CAPS headings
]


def detect_chip_family(doc_name: str) -> Optional[str]:
    """Infer chip family from filename for metadata filtering."""
    name_lower = doc_name.lower()
    families = {
        "cortex-m": ["cortex-m", "cortex_m"],
        "cortex-a": ["cortex-a", "cortex_a"],
        "risc-v": ["risc-v", "riscv", "risc_v"],
        "nvidia-gpu": ["h100", "a100", "blackwell", "hopper", "ampere"],
        "intel": ["intel", "x86", "skylake"],
        "ti-analog": ["tps", "lm", "ads", "dac"],
    }
    for family, keywords in families.items():
        if any(kw in name_lower for kw in keywords):
            return family
    return None


def is_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    line = line.strip()
    if not line or len(line) > 120:  # headings are short
        return False
    if len(line) < 3:
        return False
    return any(p.match(line) for p in HEADING_PATTERNS)


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from each page, preserving page numbers."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also extract tables and flatten them into text
            # This is crucial for datasheets where specs live in tables
            tables = page.extract_tables()
            if tables:
                table_text = "\n\n".join(
                    "\n".join(
                        " | ".join(str(cell) if cell else "" for cell in row)
                        for row in table
                    )
                    for table in tables
                )
                text = text + "\n\n[TABLE]\n" + table_text + "\n[/TABLE]"
            pages.append((i + 1, text))
    return pages


def segment_by_sections(
    pages: list[tuple[int, str]]
) -> list[tuple[str, int, str]]:
    """
    Split document into (section_title, page_num, section_text) segments
    based on detected headings.
    """
    segments = []
    current_section = "Introduction"
    current_page = 1
    current_text: list[str] = []

    for page_num, page_text in pages:
        lines = page_text.split("\n")
        for line in lines:
            if is_heading(line):
                # Flush current section
                if current_text:
                    segments.append(
                        (current_section, current_page, "\n".join(current_text))
                    )
                current_section = line.strip()
                current_page = page_num
                current_text = []
            else:
                current_text.append(line)
    # Flush final section
    if current_text:
        segments.append(
            (current_section, current_page, "\n".join(current_text))
        )
    return segments


def chunk_text(
    text: str,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    encoder: Optional[tiktoken.Encoding] = None,
) -> list[str]:
    """
    Split text into overlapping chunks sized by tokens.
    Tries to break on paragraph boundaries when possible.
    """
    if encoder is None:
        encoder = tiktoken.get_encoding("cl100k_base")

    # Split on paragraph boundaries first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(encoder.encode(para))

        # If a single paragraph exceeds target, hard-split by tokens
        if para_tokens > target_tokens:
            # Flush current chunk first
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Hard-split long paragraph
            tokens = encoder.encode(para)
            for start in range(0, len(tokens), target_tokens - overlap_tokens):
                sub = encoder.decode(tokens[start : start + target_tokens])
                chunks.append(sub)
            continue

        # Normal case: add paragraph if it fits
        if current_tokens + para_tokens <= target_tokens:
            current_chunk.append(para)
            current_tokens += para_tokens
        else:
            # Flush and start new chunk with overlap
            chunks.append("\n\n".join(current_chunk))
            # Keep last paragraph as overlap context
            if current_chunk and overlap_tokens > 0:
                last = current_chunk[-1]
                last_tokens = len(encoder.encode(last))
                if last_tokens <= overlap_tokens * 2:
                    current_chunk = [last, para]
                    current_tokens = last_tokens + para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk = [para]
                current_tokens = para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def ingest_pdf(pdf_path: Path) -> list[Chunk]:
    """Full pipeline: PDF -> list of Chunks with metadata."""
    encoder = tiktoken.get_encoding("cl100k_base")
    doc_name = pdf_path.stem
    chip_family = detect_chip_family(doc_name)

    pages = extract_pages(pdf_path)
    segments = segment_by_sections(pages)

    chunks: list[Chunk] = []
    global_idx = 0
    for section_title, page_num, section_text in segments:
        text_chunks = chunk_text(section_text, encoder=encoder)
        for i, chunk_text_content in enumerate(text_chunks):
            text_hash = hashlib.md5(chunk_text_content.encode()).hexdigest()[:8]
            chunk_id = f"{doc_name}::{global_idx}::{page_num}::{text_hash}"
            chunk_id = re.sub(r"[^\w\-:]", "_", chunk_id)
            global_idx += 1
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text_content,
                    doc_name=doc_name,
                    section=section_title,
                    page_num=page_num,
                    token_count=len(encoder.encode(chunk_text_content)),
                    chip_family=chip_family,
                )
            )
    return chunks


def ingest_all(raw_dir: Path, output_path: Path) -> list[Chunk]:
    """Process all PDFs in raw_dir and save chunks to JSON."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {raw_dir}")

    all_chunks: list[Chunk] = []
    for pdf_path in tqdm(pdf_paths, desc="Ingesting PDFs"):
        try:
            chunks = ingest_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"  {pdf_path.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  FAILED {pdf_path.name}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    print(f"\nTotal: {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
    print(f"Saved to {output_path}")
    return all_chunks


if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    output_path = Path(__file__).parent.parent / "data" / "processed" / "chunks.json"
    ingest_all(raw_dir, output_path)
