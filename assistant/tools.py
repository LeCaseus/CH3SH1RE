import requests
from ddgs import DDGS

def search_web(query):
    """Use duckduckgo_search library instead of HTML scraping for more reliable results."""
    try:
        results = DDGS().text(query, max_results=3)
        
        if not results:
            return f"No quality results found for '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n"
                f"   {r['body']}\n"
                f"   ({r['href']})"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Search error: {str(e)}"


def analyze_image(image_path):
    # Calls the separate vision service running on port 5000
    # Start vision_service.py separately before using image analysis
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://127.0.0.1:5000/analyze', files=files, timeout=10)
            data = response.json()
            return data.get('description', 'Unable to analyze image')
    except Exception as e:
        return f"Image analysis error: {str(e)}"
