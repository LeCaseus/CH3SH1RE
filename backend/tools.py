import urllib.request
import re
from ddgs import DDGS


def fetch_page(url: str, max_chars: int = 3000) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"Fetch error: {str(e)}"


def search_web(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=3)

        if not results:
            return f"No results found for '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            full = fetch_page(r["href"])
            body = full if "Fetch error" not in full else r["body"]
            formatted.append(f"{i}. {r['title']}\n" f"   {body}\n" f"   ({r['href']})")
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"


def read_file(file_path: str) -> str:
    try:
        if file_path.endswith(".pdf"):
            import fitz

            doc = fitz.open(file_path)
            return "\n".join(str(doc[i].get_text()) for i in range(len(doc)))

        elif file_path.endswith(".docx"):
            from docx import Document

            doc = Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        elif file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        else:
            return "Unsupported file type. Please upload a PDF, DOCX, or TXT file."

    except Exception as e:
        return f"File read error: {str(e)}"
