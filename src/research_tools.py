from typing import List, Dict, Optional
import os, re, time
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ===== DuckDuckGo Search  =====
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    print(" duckduckgo-search not installed. Install with: pip install duckduckgo-search")

import wikipedia


def _build_session(
    user_agent: str = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)",
) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

session = _build_session()


# ===== PDF Utilities (unchanged) =====
def ensure_pdf_url(abs_or_pdf_url: str) -> str:
    url = abs_or_pdf_url.strip().replace("http://", "https://")
    if "/pdf/" in url and url.endswith(".pdf"):
        return url
    url = url.replace("/abs/", "/pdf/")
    if not url.endswith(".pdf"):
        url += ".pdf"
    return url

def clean_text(s: str) -> str:
    s = re.sub(r"-\n", "", s)
    s = re.sub(r"\r\n|\r", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def fetch_pdf_bytes(pdf_url: str, timeout: int = 90) -> bytes:
    r = session.get(pdf_url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.content

def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    try:
        import fitz
        out = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n = len(doc)
            limit = n if max_pages is None else min(max_pages, n)
            for i in range(limit):
                out.append(doc.load_page(i).get_text("text"))
        return "\n".join(out)
    except Exception:
        pass
    
    try:
        from pdfminer.high_level import extract_text_to_fp
        buf_in = BytesIO(pdf_bytes)
        buf_out = BytesIO()
        extract_text_to_fp(buf_in, buf_out)
        return buf_out.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")


# ===== arXiv Search (unchanged) =====
def arxiv_search_tool(query: str, max_results: int = 3) -> List[Dict]:
    _INCLUDE_PDF = True
    _EXTRACT_TEXT = True
    _MAX_PAGES = 6
    _TEXT_CHARS = 5000
    _SLEEP_SECONDS = 1.0

    api_url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(query)}&start=0&max_results={max_results}"
    )

    out: List[Dict] = []
    try:
        resp = session.get(api_url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": f"arXiv API request failed: {e}"}]

    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
            url_abs = entry.findtext("atom:id", default="", namespaces=ns) or ""
            abstract_summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()

            authors = []
            for a in entry.findall("atom:author", ns):
                nm = a.findtext("atom:name", default="", namespaces=ns)
                if nm:
                    authors.append(nm)

            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break
            if not link_pdf and url_abs:
                link_pdf = ensure_pdf_url(url_abs)

            item = {
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abs,
                "summary": abstract_summary,
                "link_pdf": link_pdf,
            }

            pdf_bytes = None
            if (_INCLUDE_PDF or _EXTRACT_TEXT) and link_pdf:
                try:
                    pdf_bytes = fetch_pdf_bytes(link_pdf, timeout=90)
                    time.sleep(_SLEEP_SECONDS)
                except Exception as e:
                    item["pdf_error"] = f"PDF fetch failed: {e}"

            if _EXTRACT_TEXT and pdf_bytes:
                try:
                    text = pdf_bytes_to_text(pdf_bytes, max_pages=_MAX_PAGES)
                    text = clean_text(text) if text else ""
                    if text:
                        item["summary"] = text[:_TEXT_CHARS]
                except Exception as e:
                    item["text_error"] = f"Text extraction failed: {e}"

            out.append(item)
        return out
    except Exception as e:
        return [{"error": f"Unexpected error: {e}"}]


# ===== DuckDuckGo Search  =====
def duckduckgo_search_tool(
    query: str, 
    max_results: int = 5
) -> List[Dict]:
    """
    FREE web search using DuckDuckGo
    
    Args:
        query: Search query string
        max_results: Number of results to return (default 5)
    
    Returns:
        List of dicts with keys: title, content, url
    """
    if not DUCKDUCKGO_AVAILABLE:
        return [{"error": "duckduckgo-search not installed"}]
    
    try:
        results = []
        with DDGS() as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                if i >= max_results:
                    break
                results.append({
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "url": r.get("href", "")
                })
                time.sleep(0.3)  # Rate limiting
        return results
    except Exception as e:
        return [{"error": f"DuckDuckGo search failed: {str(e)}"}]


# ===== Wikipedia Search (unchanged) =====
def wikipedia_search_tool(query: str, sentences: int = 5) -> List[Dict]:
    try:
        page_title = wikipedia.search(query)[0]
        page = wikipedia.page(page_title)
        summary = wikipedia.summary(page_title, sentences=sentences)
        return [{"title": page.title, "summary": summary, "url": page.url}]
    except Exception as e:
        return [{"error": str(e)}"]


# ===== Tool Definitions =====
arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches arXiv for academic papers (CS, Math, Physics, Stats only)",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "max_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
}

duckduckgo_tool_def = {
    "type": "function",
    "function": {
        "name": "duckduckgo_search_tool",
        "description": "FREE general web search using DuckDuckGo",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches Wikipedia for encyclopedia articles",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to search"},
                "sentences": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}

# Tool mapping for execution
tool_mapping = {
    "arxiv_search_tool": arxiv_search_tool,
    "duckduckgo_search_tool": duckduckgo_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}
