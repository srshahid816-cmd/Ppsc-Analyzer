import os
import re
import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import Counter

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="PPSC Examiner AI", page_icon="🎯", layout="wide")

BASE = Path("data")
PAPER_DIR = BASE / "papers"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
HEADERS = {"User-Agent": "Mozilla/5.0 (PPSC-Examiner-AI/2.0)"}

# Public archives. These are third-party sources, so the app labels their provenance.
SOURCES = [
    ("TestPoint", "https://testpointpk.com/past-papers-mcqs/ppsc-past-papers?page=1"),
    ("GK360", "https://gk360.pk/papers/"),
    ("SawalNama", "https://www.sawalnamappsc.pro/pastpapers"),
    ("PaperMCQs", "https://papermcqs.online/past-papers"),
]

YEAR_RE = re.compile(r"\b(20(?:2[1-9]|1[0-9]))\b")
STOPWORDS = set(
    "the and for are was were with from into this that have has had ppsc punjab paper "
    "questions question officer post department past solved mcqs test year".split()
)


def get(url: str):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:130]


def extract_year(text: str):
    years = [int(x) for x in YEAR_RE.findall(text or "")]
    return max(years) if years else 0


def relevance_score(title: str, url: str, job: str, source: str) -> int:
    hay = f"{title} {url}".lower()
    terms = [x.lower() for x in re.findall(r"[a-z0-9]+", job) if len(x) > 2]
    score = sum(3 for term in terms if term in hay)
    if "ppsc" in hay:
        score += 2
    if "paper" in hay or "mcq" in hay:
        score += 2
    if "past" in hay or "solved" in hay:
        score += 1
    year = extract_year(hay)
    if year >= 2026:
        score += 8
    elif year == 2025:
        score += 6
    elif year == 2024:
        score += 4
    elif year >= 2021:
        score += 2
    # Small source-quality preference; not an assertion that these are official.
    if source == "TestPoint":
        score += 2
    elif source in {"GK360", "SawalNama"}:
        score += 1
    return score


def discover(job: str):
    hits = []
    errors = []

    for source, base in SOURCES:
        try:
            soup = BeautifulSoup(get(base).text, "html.parser")
            links = soup.find_all("a", href=True)
            for a in links:
                title = clean(a.get_text(" ", strip=True))
                url = urljoin(base, a["href"])
                if not title or url.startswith("javascript:"):
                    continue
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"}:
                    continue
                hay = f"{title} {url}".lower()
                # Only retain links that plausibly belong to PPSC/past-paper material.
                if not any(k in hay for k in ("ppsc", "past-paper", "past_paper", "mcq", "solved")):
                    continue
                score = relevance_score(title, url, job, source)
                hits.append(
                    {
                        "source": source,
                        "title": title,
                        "url": url,
                        "year": extract_year(hay),
                        "score": score,
                    }
                )
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    unique = {}
    for item in hits:
        old = unique.get(item["url"])
        if old is None or item["score"] > old["score"]:
            unique[item["url"]] = item

    ranked = sorted(
        unique.values(), key=lambda x: (-x["score"], -x["year"], len(x["title"]))
    )
    return ranked[:80], errors


def save_url(item: dict, job: str, index: int):
    url = item["url"]
    is_pdf = ".pdf" in url.lower()
    suffix = ".pdf" if is_pdf else ".html"
    path = PAPER_DIR / safe(f"{job}_{index}_{item['title']}")
    path = path.with_suffix(suffix)

    if not path.exists():
        r = get(url)
        path.write_bytes(r.content if is_pdf else r.text.encode("utf-8", errors="ignore"))

    meta = {
        "source": item["source"],
        "title": item["title"],
        "url": url,
        "job": job,
        "year": item.get("year", 0),
        "score": item.get("score", 0),
        "saved": time.strftime("%Y-%m-%d"),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def text_from(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return clean("\n".join((page.extract_text() or "") for page in reader.pages))

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return clean(soup.get_text(" ", strip=True))


def download_hits(hits, job):
    saved = []
    for i, item in enumerate(hits):
        try:
            meta = save_url(item, job, i)
            saved.append(meta)
        except Exception:
            continue
    return saved


def corpus():
    out = []
    for path in PAPER_DIR.iterdir():
        if path.suffix.lower() not in {".pdf", ".html"}:
            continue
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            text = text_from(path)
            if len(text) > 250:
                out.append({**meta, "file": path.name, "text": text})
        except Exception:
            continue
    return out


def topic_counts(papers):
    counts = Counter()
    for paper in papers:
        words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", paper["text"].lower())
        counts.update(word for word in words if word not in STOPWORDS)
    return counts


def build_schema():
    question_properties = {
        "id": {"type": "integer"},
        "question": {"type": "string"},
        "options": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "explanation": {"type": "string"},
        "subject": {"type": "string"},
        "topic": {"type": "string"},
        "difficulty": {"type": "string", "enum": ["Easy", "Medium", "Hard"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "type": {
            "type": "string",
            "enum": ["RepeatedConcept", "PatternTransfer", "NewConcept"],
        },
        "source_basis": {"type": "string"},
    }
    question_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": question_properties,
        "required": list(question_properties.keys()),
    }
    paper_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "paper_title": {"type": "string"},
            "questions": {
                "type": "array",
                "items": question_schema,
            },
        },
        "required": ["paper_title", "questions"],
    }
    return {
        "name": "ppsc_paper",
        "strict": True,
        "schema": paper_schema,
    }


def generate(job, papers, number, instruction, model):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it in your GitHub/Streamlit secrets or environment variables."
        )

    client = OpenAI(api_key=api_key)

    # Latest/highest-relevance items first. Limit corpus to control token usage.
    selected = sorted(
        papers,
        key=lambda p: (-int(p.get("year", 0)), -int(p.get("score", 0)), -len(p["text"])),
    )[:18]
    corpus_text = "\n\n--- SOURCE ---\n".join(
        f"TITLE: {p['title']}\nSOURCE: {p['source']}\nYEAR: {p.get('year', 0)}\n"
        f"URL: {p['url']}\nTEXT: {p['text'][:10000]}"
        for p in selected
    )

    prompt = f"""You are a PPSC teacher AND examiner simulator.

Target job/post: {job}
Generate exactly {number} original MCQs.

Examiner instruction:
{instruction}

Method:
1. Give the latest available papers more weight, especially 2026, then 2025, then older years.
2. Give same-post/same-department evidence more weight than unrelated posts.
3. Identify repeated concepts, recurring subject areas, wording/style patterns, and gaps.
4. Build a balanced paper rather than simply copying frequent questions.
5. Questions must be answerable from established knowledge; do not invent facts or citations.
6. Do not copy long passages from sources.
7. Do NOT claim that any question is guaranteed to appear in a future PPSC paper.
8. confidence means strength of evidence supporting the question's inclusion, NOT probability of appearing.
9. Avoid duplicate questions and ambiguous options.

Each question must contain:
- id
- question
- exactly 4 options
- answer as A/B/C/D
- short explanation
- subject
- topic
- difficulty: Easy/Medium/Hard
- confidence: 0-100
- type: RepeatedConcept, PatternTransfer, or NewConcept
- source_basis: brief description of which source/year/concept influenced it

PAST-PAPER CORPUS:
{corpus_text}
"""

    response = client.chat.completions.create(
        model=model,
        temperature=0.25,
        response_format={
            "type": "json_schema",
            "json_schema": build_schema(),
        },
        messages=[
            {"role": "system", "content": "Return only valid JSON matching the supplied schema."},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


st.title("🎯 PPSC Examiner AI — Auto Past-Paper Engine")
st.caption(
    "Enter a job/post → automatically discover public past-paper pages → build a local corpus → analyze → generate a fresh mock paper."
)

with st.sidebar:
    st.header("Exam Settings")
    job = st.text_input("Target PPSC job/post", "PERA Enforcement Officer")
    number = st.selectbox("Questions", [25, 50, 100], index=2)
    model = st.text_input("AI model", os.getenv("OPENAI_MODEL", "gpt-5.6-mini"))
    instruction = st.text_area(
        "Examiner instruction",
        "Prioritize latest pattern, same-post papers, repeated concepts, syllabus relevance, difficulty balance, and plausible new examiner-style questions.",
    )

fetch_tab, lab_tab, paper_tab, dashboard_tab = st.tabs(
    ["🌐 Auto Fetch", "📊 Pattern Lab", "📝 Paper Maker", "🎯 Dashboard"]
)

with fetch_tab:
    st.info("No manual PDF upload is required. The app searches the configured public PPSC archives automatically.")
    if st.button("🔎 Automatically Find Past Papers", type="primary"):
        with st.spinner("Searching public PPSC archives..."):
            hits, errors = discover(job)
            saved = download_hits(hits, job)
        st.session_state["hits"] = hits
        st.session_state["saved"] = saved
        st.success(f"Found {len(hits)} candidate links; saved {len(saved)} readable items.")
        if hits:
            st.dataframe(
                pd.DataFrame(hits)[["source", "title", "year", "score", "url"]],
                use_container_width=True,
                hide_index=True,
            )
        if errors:
            st.warning("Some sources were unavailable: " + " | ".join(errors))

    st.metric("Local corpus items", len(corpus()))

with lab_tab:
    papers = corpus()
    if not papers:
        st.info("Run Auto Fetch first.")
    else:
        counts = topic_counts(papers)
        c1, c2, c3 = st.columns(3)
        c1.metric("Corpus items", len(papers))
        c2.metric("Total words", sum(len(p["text"].split()) for p in papers))
        c3.metric("2026+ items", sum(int(p.get("year", 0)) >= 2026 for p in papers))
        st.bar_chart(
            pd.DataFrame(counts.most_common(30), columns=["Term", "Frequency"]).set_index("Term")
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Year": p.get("year", 0),
                        "Title": p["title"],
                        "Source": p["source"],
                        "Score": p.get("score", 0),
                        "URL": p["url"],
                    }
                    for p in papers
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

with paper_tab:
    papers = corpus()
    if not papers:
        st.info("Run Auto Fetch first.")
    elif st.button("🚀 Generate New Paper", type="primary"):
        with st.spinner("Analyzing the corpus and simulating the examiner..."):
            try:
                st.session_state["paper"] = generate(job, papers, number, instruction, model)
            except Exception as exc:
                st.error(str(exc))

    if "paper" in st.session_state:
        paper_data = st.session_state["paper"]
        questions = paper_data["questions"]
        st.success(f"Generated {len(questions)} MCQs.")
        for q in questions:
            with st.expander(
                f"Q{q['id']} • {q['subject']} • {q['difficulty']} • Evidence {q['confidence']}%"
            ):
                st.markdown(f"**{q['question']}**")
                for letter, option in zip("ABCD", q["options"]):
                    st.write(f"**{letter}.** {option}")
                st.write(f"**Answer:** {q['answer']}")
                st.write(f"**Explanation:** {q['explanation']}")
                st.caption(
                    f"{q['type']} | {q['topic']} | {q['source_basis']}"
                )

with dashboard_tab:
    if "paper" not in st.session_state:
        st.info("Generate a paper first.")
    else:
        df = pd.DataFrame(st.session_state["paper"]["questions"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evidence ≥80", int((df["confidence"] >= 80).sum()))
        c2.metric("Repeated concept", int((df["type"] == "RepeatedConcept").sum()))
        c3.metric("Pattern transfer", int((df["type"] == "PatternTransfer").sum()))
        c4.metric("New concept", int((df["type"] == "NewConcept").sum()))
        st.bar_chart(df.groupby("subject").size())
        st.info(
            "Evidence confidence is not a guarantee that a question will appear in a future PPSC paper."
        )
