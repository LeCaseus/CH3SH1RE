from ddgs import DDGS


def search_web(query: str) -> str:
    # your original DDG logic — already tested and working
    try:
        results = DDGS().text(query, max_results=3)

        if not results:
            return f"No results found for '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n" f"   {r['body']}\n" f"   ({r['href']})"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"


def read_file(file_path: str) -> str:
    # reads PDF or Word files from the uploads folder
    try:
        if file_path.endswith(".pdf"):
            import fitz

            doc = fitz.open(file_path)
            return "\n".join(str(page.get_text()) for page in doc)

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
