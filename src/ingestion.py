import os
import re
import uuid
from typing import List


CS50_NOTES_URLS = {
    "Functions and Variables":     "https://cs50.harvard.edu/python/notes/0/",
    "Conditionals":                "https://cs50.harvard.edu/python/notes/1/",
    "Loops":                       "https://cs50.harvard.edu/python/notes/2/",
    "Exceptions":                  "https://cs50.harvard.edu/python/notes/3/",
    "Libraries":                   "https://cs50.harvard.edu/python/notes/4/",
    "Unit Tests":                  "https://cs50.harvard.edu/python/notes/5/",
    "File I/O":                    "https://cs50.harvard.edu/python/notes/6/",
    "Regular Expressions":         "https://cs50.harvard.edu/python/notes/7/",
    "Object-Oriented Programming": "https://cs50.harvard.edu/python/notes/8/",
    "Et Cetera":                   "https://cs50.harvard.edu/python/notes/9/",
}

NOISE_PATTERNS = [
    re.compile(r"interested in.*?certificate", re.I),
    re.compile(r"donate", re.I),
    re.compile(r"zoom meetings", re.I),
    re.compile(r"cs50\.ai", re.I),
    re.compile(r"ed discussion", re.I),
    re.compile(r"apple tv|edx|freecodecamp|google tv|roku|youtube", re.I),
    re.compile(r"status page", re.I),
    re.compile(r"license", re.I),
    re.compile(r"communities|bluesky|clubhouse|discord|instagram|linkedin|medium|quora|reddit|slack|snapchat|soundcloud|stack exchange|telegram|tiktok|threads", re.I),
]


def fetch_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    print(f"  Fetching: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    response.encoding = "utf-8"
    text = response.text
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-")

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        main = soup

    lines = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
        text = tag.get_text(separator=" ", strip=True)
        if not text or len(text) < 5:
            continue

        if any(p.search(text) for p in NOISE_PATTERNS):
            continue

        if tag.name in ["h1", "h2", "h3", "h4"]:
            lines.append(f"\n# {text}\n")
        elif tag.name == "pre":
            lines.append(f"\n{text}\n")
        else:
            lines.append(text)

    raw = "\n".join(lines)

    lecture_start = re.search(r"#\s*Lecture \d+", raw)
    if lecture_start:
        raw = raw[lecture_start.start():]

    return raw.strip()


def load_text(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        return fetch_url(source)

    if os.path.isfile(source):
        if source.endswith(".pdf"):
            import PyPDF2
            with open(source, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
        with open(source, "r", encoding="utf-8") as f:
            return f.read()

    return source


def recursive_chunk(text: str, chunk_word_count: int = 800, overlap: int = 120) -> List[dict]:
    heading_pattern = re.compile(r"(?m)^#{1,4} .+")
    sections = []
    last_heading = "Introduction"
    last_pos = 0

    for match in heading_pattern.finditer(text):
        if match.start() > last_pos:
            sections.append({
                "heading": last_heading,
                "text": text[last_pos:match.start()].strip()
            })
        last_heading = re.sub(r"^#+\s*", "", match.group()).strip()
        last_pos = match.end()

    if last_pos < len(text):
        sections.append({"heading": last_heading, "text": text[last_pos:].strip()})

    chunks = []
    chunk_id = 0
    for section in sections:
        body = section["text"]
        if not body or len(body.split()) < 10:
            continue
        words = body.split()
        i = 0
        while i < len(words):
            window = words[i:i + chunk_word_count]
            chunk_text = " ".join(window)
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "index": chunk_id,
                "section_heading": section["heading"],
                "topic": "Python Fundamentals",
                "subtopic": section["heading"],
                "text": chunk_text,
                "artifact_targets": ["notes", "quiz", "slides", "takeaways"]
            })
            chunk_id += 1
            i += chunk_word_count - overlap

    return chunks