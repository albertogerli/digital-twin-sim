"""Document processing for RAG-grounded simulations.

Extracts text from uploaded files (PDF, DOCX, TXT, MD, JSON)
and injects it into the simulation pipeline as additional context.

Two modes:
1. Raw context: text extracted → injected into brief_analyzer as context
2. Structured seed data: JSON → parsed as VerifiedStakeholder/VerifiedDemographic
"""

import json
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"


def ensure_upload_dir(sim_id: str) -> Path:
    """Create and return the upload directory for a simulation."""
    d = UPLOAD_DIR / sim_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_uploaded_file(sim_id: str, filename: str, content: bytes) -> Path:
    """Save an uploaded file and return its path."""
    d = ensure_upload_dir(sim_id)
    safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in filename)
    path = d / safe_name
    path.write_bytes(content)
    logger.info(f"Saved upload: {path} ({len(content)} bytes)")
    return path


def extract_text(file_path: Path) -> str:
    """Extract text from a file. Supports PDF, DOCX, TXT, MD, JSON."""
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md", ".csv"):
        return file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".json":
        return file_path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _extract_pdf(file_path)

    if suffix in (".docx", ".doc"):
        return _extract_docx(file_path)

    # Fallback: try as text
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.warning(f"Cannot extract text from {file_path}")
        return ""


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF using PyPDF2 or pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
        return "\n\n".join(text_parts)
    except ImportError:
        logger.warning("No PDF library available. Install pdfplumber or PyPDF2.")
        return f"[PDF file: {path.name} — install pdfplumber to extract text]"


def _extract_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx."""
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        logger.warning("python-docx not installed. Install with: pip install python-docx")
        return f"[DOCX file: {path.name} — install python-docx to extract text]"


def process_uploads(sim_id: str, rag_store=None) -> dict:
    """Process all uploaded files for a simulation.

    If `rag_store` (RAGStore instance) is provided, each file is also chunked,
    embedded, and added to the store so agents can retrieve grounded snippets
    at round time.

    Returns:
        {
            "context_text": str,       # Combined text from all documents
            "seed_data_path": str,     # Path to seed data dir (if JSON stakeholders found)
            "file_count": int,
            "total_chars": int,
            "rag_chunks": int,         # Total chunks indexed in rag_store
        }
    """
    upload_dir = UPLOAD_DIR / sim_id
    if not upload_dir.exists():
        return {"context_text": "", "seed_data_path": "", "file_count": 0, "total_chars": 0, "rag_chunks": 0}

    context_parts = []
    seed_data_path = ""
    file_count = 0
    doc_index = 0  # for stable doc_ids (d1, d2, …) across the simulation

    for file_path in sorted(upload_dir.iterdir()):
        if file_path.name.startswith("."):
            continue

        text = extract_text(file_path)
        if not text:
            continue

        file_count += 1
        doc_index += 1
        doc_id = f"d{doc_index}"

        # Check if it's structured seed data (JSON with stakeholders)
        if file_path.suffix == ".json":
            seed_path = _try_parse_seed_data(file_path, sim_id)
            if seed_path:
                seed_data_path = seed_path
                continue  # Don't add raw JSON to context text

        # Add as context with file header
        context_parts.append(
            f"--- DOCUMENTO: {file_path.name} ---\n{text}"
        )

        # ── RAG: chunk + embed for retrieval-grounded reasoning ──
        if rag_store is not None:
            try:
                rag_store.add_document(doc_id=doc_id, title=file_path.name, text=text)
            except Exception as exc:
                logger.warning(f"RAG indexing failed for {file_path.name}: {exc}")

    context_text = "\n\n".join(context_parts)

    return {
        "context_text": context_text,
        "seed_data_path": seed_data_path,
        "file_count": file_count,
        "total_chars": len(context_text),
        "rag_chunks": rag_store.chunk_count if rag_store is not None else 0,
    }


def _try_parse_seed_data(json_path: Path, sim_id: str) -> str:
    """Try to parse a JSON file as structured seed data.

    Accepts formats:
    1. {"stakeholders": [...], "demographics": [...], "context": "..."}
    2. [{"name": "...", "role": "...", "position": ...}, ...]  (stakeholders array)

    Returns seed_data directory path if successful, empty string otherwise.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return ""

    stakeholders = None
    demographics = None
    context_text = ""

    if isinstance(data, dict):
        stakeholders = data.get("stakeholders")
        demographics = data.get("demographics")
        context_text = data.get("context", data.get("context_text", ""))
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        # Check if it looks like stakeholders
        if "name" in data[0] and ("role" in data[0] or "position" in data[0]):
            stakeholders = data

    if not stakeholders:
        return ""

    # Build seed data directory
    seed_dir = UPLOAD_DIR / sim_id / "seed_data"
    seed_dir.mkdir(exist_ok=True)

    # Write stakeholders
    with open(seed_dir / "stakeholders.json", "w") as f:
        json.dump(stakeholders, f, indent=2, ensure_ascii=False)

    if demographics:
        with open(seed_dir / "demographics.json", "w") as f:
            json.dump(demographics, f, indent=2, ensure_ascii=False)

    if context_text:
        (seed_dir / "context.md").write_text(context_text, encoding="utf-8")

    logger.info(f"Parsed seed data: {len(stakeholders)} stakeholders, "
                f"{len(demographics or [])} demographics")
    return str(seed_dir)


def cleanup_uploads(sim_id: str):
    """Remove uploaded files after simulation completes."""
    upload_dir = UPLOAD_DIR / sim_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
