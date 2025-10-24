import os
import re
import time
import json
import traceback
import urllib.parse
from datetime import datetime
import httpx

# keep your existing imports (QdrantClient, SentenceTransformer, etc.)
# ... (they are already present earlier in your module) ...

# ---------- New helpers: arXiv search + PDF download ----------

def _sanitize_filename(s: str, max_len: int = 120) -> str:
    # simple sanitizer for titles -> filesystem-safe
    s = s.strip()
    s = re.sub(r'[\s]+', '_', s)
    s = re.sub(r'[^0-9A-Za-z_\-\.]', '', s)
    return s[:max_len]

def search_arxiv_for_papers(query: str, max_results: int = 10, timeout: int = 15) -> list:
    """
    Query arXiv API and return a list of dicts: {"title": ..., "id": ..., "pdf_url": ...}
    """
    q = urllib.parse.quote_plus(query)
    api_url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
    try:
        resp = httpx.get(api_url, timeout=timeout)
        resp.raise_for_status()
        xml = resp.text
    except Exception as e:
        print(f"[WARN] arXiv query failed for '{query}': {e}")
        return []

    # parse Atom feed (basic)
    try:
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml)
        entries = root.findall("atom:entry", ns)
        results = []
        for entry in entries:
            title_el = entry.find("atom:title", ns)
            id_el = entry.find("atom:id", ns)
            if title_el is None or id_el is None:
                continue
            title = (title_el.text or "").strip().replace("\n", " ")
            id_text = (id_el.text or "").strip()
            # id_text looks like "http://arxiv.org/abs/2101.00001v1" -> get last part
            paper_id = id_text.rsplit("/", 1)[-1]
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            results.append({"title": title, "id": paper_id, "pdf_url": pdf_url})
        return results
    except Exception as e:
        print(f"[WARN] Failed to parse arXiv feed for '{query}': {e}")
        return []

def download_pdf_to_path(pdf_url: str, dest_path: str, timeout: int = 60) -> bool:
    """
    Stream-download a PDF at pdf_url to dest_path. Returns True on success.
    """
    headers = {"User-Agent": "rag-ingest-bot/1.0 (+https://yourapp.example)"}
    try:
        with httpx.stream("GET", pdf_url, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            # ensure parent dir exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_bytes(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        # very small sanity check: file must be > 1 KB
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
            return True
        else:
            print(f"[WARN] downloaded file {dest_path} is suspiciously small ({os.path.getsize(dest_path) if os.path.exists(dest_path) else 0} bytes)")
            return False
    except Exception as e:
        print(f"[ERROR] failed to download {pdf_url}: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        return False

# ---------- Updated search_and_ingest_papers (no dedupe) ----------

def search_and_ingest_papers(query: str, max_papers: int = 10) -> str:
    """
    Search arXiv for `query`, download the top `max_papers` PDFs, and ingest them all into Qdrant.
    This function intentionally DOES NOT deduplicate: every found PDF will be downloaded and ingested.
    """
    print(f"[INFO] Starting real ingestion for query: '{query}'", flush=True)

    # 1) Initialize clients & helpers
    client = QdrantClient(url=QDRANT_URL, api_key=(QDRANT_API_KEY.strip() if QDRANT_API_KEY else None))
    emb_gen = EmbeddingGenerator(model_name=EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    tokenizer = TokenizerWrapper(encoding_name=TOKENIZER_NAME)

    os.makedirs(PDF_DIR, exist_ok=True)

    # 2) Discover papers via arXiv
    found = search_arxiv_for_papers(query, max_results=max_papers)
    if not found:
        return f"No papers found on arXiv for query '{query}'."

    print(f"[INFO] arXiv returned {len(found)} papers for query '{query}'.", flush=True)

    # 3) Download each PDF, collect file paths
    downloaded_paths = []
    for item in found:
        title = item.get("title") or item.get("id")
        paper_id = item.get("id")
        pdf_url = item.get("pdf_url")
        safe_title = _sanitize_filename(title, max_len=80)
        filename = f"{safe_title[:80]}_{paper_id}.pdf"
        filepath = os.path.join(PDF_DIR, filename)
        try:
            ok = download_pdf_to_path(pdf_url, filepath)
            if ok:
                downloaded_paths.append(filepath)
                print(f"[INFO] Downloaded {pdf_url} -> {filepath}", flush=True)
            else:
                print(f"[WARN] Skipping ingestion for {pdf_url}: download failed or file too small.", flush=True)
        except Exception as e:
            print(f"[ERROR] Exception downloading {pdf_url}: {e}", flush=True)
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

    if not downloaded_paths:
        return f"No PDFs downloaded successfully for query '{query}'. Nothing to ingest."

    # 4) Ingest each downloaded PDF
    total_chunks_added = 0
    ingested_papers = 0
    for pdf_path in downloaded_paths:
        filename = os.path.basename(pdf_path)
        print(f"[INFO] Starting ingestion for downloaded paper: {filename}", flush=True)
        t0 = time.time()
        try:
            chunks_added = _ingest_pdf(pdf_path, client, emb_gen, tokenizer)
            total_chunks_added += chunks_added
            ingested_papers += 1
            # cleanup local PDF
            try:
                os.remove(pdf_path)
            except Exception as e:
                print(f"[WARN] failed to delete local PDF {pdf_path}: {e}", flush=True)
            print(f"[INFO] Ingestion finished for {filename}. Added {chunks_added} chunks in {time.time()-t0:.2f}s.", flush=True)
        except Exception as e:
            print(f"[ERROR] Ingestion failed for {filename}: {e}", flush=True)
            print(traceback.format_exc())
            # attempt cleanup
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
            # continue with next file (no early return) so failing one paper won't stop the whole batch

    return f"Successfully downloaded and ingested {ingested_papers} paper(s) for query '{query}', adding {total_chunks_added} chunks to Qdrant."
