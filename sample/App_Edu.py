import os
import streamlit as st
import streamlit.components.v1 as components

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="ERGIT | Education Resource-Gap Intelligence Tool",
    page_icon="🎓",
    layout="wide"
)


# --------------------------------------------------
# 2. Design helpers
# --------------------------------------------------

def html(block):
    """Render HTML without indentation, so Streamlit does not
    treat it as a code block."""
    cleaned = "\n".join(line.strip() for line in block.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


LOGO_SVG = """
<svg width="46" height="46" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#2563EB"/><stop offset="1" stop-color="#1E3A8A"/>
</linearGradient>
</defs>
<rect x="1" y="1" width="46" height="46" rx="13" fill="url(#lg)"/>
<rect x="14" y="13" width="4" height="9" rx="1.5" fill="#FACC15"/>
<rect x="21" y="9" width="4" height="13" rx="1.5" fill="#FACC15"/>
<rect x="28" y="15" width="4" height="7" rx="1.5" fill="#34D399"/>
<path d="M8 26 C14 23 20 23 24 27 C28 23 34 23 40 26 L40 38 C34 35 28 35 24 39 C20 35 14 35 8 38 Z" fill="#FFFFFF"/>
<path d="M24 27 L24 39" stroke="#1E3A8A" stroke-width="1.4"/>
</svg>
"""


def inject_css():
    html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
--ink: #0F1E4A;
--blue: #2563EB;
--navy: #1E3A8A;
--yellow: #FACC15;
--green: #10B981;
--coral: #F97066;
--muted: #5B6680;
--line: #DCE6F5;
}

/* Exercise-book paper: ruled blue lines */
[data-testid="stAppViewContainer"] {
background-color: #FBFCFE;
background-image: linear-gradient(var(--line) 1px, transparent 1px);
background-size: 100% 34px;
}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stDecoration"] {display: none;}
#MainMenu, footer {visibility: hidden;}

.block-container {
max-width: 1120px;
padding-top: 1.4rem !important;
padding-bottom: 6rem !important;
position: relative;
}
/* Red margin line of an exercise book */
.block-container::before {
content: ""; position: absolute; top: 0; bottom: 0; left: -14px;
width: 2px; background: rgba(249, 112, 102, 0.45);
}

html, body, p, li, label, .stMarkdown {
font-family: 'Inter', sans-serif; color: var(--ink);
}
h1, h2, h3, .jakarta {font-family: 'Plus Jakarta Sans', sans-serif !important; color: var(--ink);}

/* Top bar with logo on the right */
.topbar {
display: flex; align-items: center; justify-content: space-between;
margin-bottom: 1.2rem;
}
.pill {
display: inline-flex; align-items: center; gap: 0.45rem;
background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
padding: 0.4rem 0.9rem; font-size: 0.85rem; color: var(--muted);
box-shadow: 0 2px 8px rgba(15, 30, 74, 0.05);
}
.dot {width: 8px; height: 8px; border-radius: 50%; background: var(--green);
box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.15);}
.brand {display: flex; align-items: center; gap: 0.7rem; text-align: right;}
.brand-name {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 1.45rem; letter-spacing: 0.06em; color: var(--navy); line-height: 1;
}
.brand-sub {font-size: 0.7rem; color: var(--muted); letter-spacing: 0.04em; margin-top: 0.2rem;}

/* Section titles */
.eyebrow {
font-size: 0.78rem; font-weight: 600; letter-spacing: 0.14em;
text-transform: uppercase; color: var(--blue); margin-bottom: 0.35rem;
}
.section-title {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 1.9rem; color: var(--ink); margin: 0 0 0.4rem 0; line-height: 1.2;
}
.section-title .hl {
background: linear-gradient(transparent 62%, rgba(250, 204, 21, 0.55) 62%);
padding: 0 0.15rem;
}
.lead {font-size: 1.02rem; color: var(--muted); line-height: 1.65; max-width: 720px;}
.gap {height: 2.6rem;}

/* Stat cards with progress bars */
.stat {
background: #FFFFFF; border-radius: 18px; padding: 1.3rem 1.3rem 1.1rem;
border: 1px solid var(--line); box-shadow: 0 10px 30px rgba(15, 30, 74, 0.07);
height: 100%;
}
.stat-top {display: flex; justify-content: space-between; align-items: center;}
.stat-icon {
width: 40px; height: 40px; border-radius: 12px; display: flex;
align-items: center; justify-content: center; font-size: 1.25rem;
}
.stat-num {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 2.2rem; color: var(--ink); margin-top: 0.8rem; line-height: 1;
}
.stat-num small {font-size: 1rem; color: var(--muted); font-weight: 600;}
.stat-label {font-weight: 600; margin-top: 0.35rem; font-size: 0.95rem;}
.bar {height: 8px; background: #EEF2F9; border-radius: 99px; margin-top: 0.9rem; overflow: hidden;}
.bar > span {display: block; height: 100%; border-radius: 99px;}
.stat-note {font-size: 0.78rem; color: var(--muted); margin-top: 0.55rem;}

/* Feature cards */
.feature {
background: #FFFFFF; border-radius: 18px; padding: 1.5rem;
border: 1px solid var(--line); height: 100%;
transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature:hover {transform: translateY(-4px); box-shadow: 0 16px 36px rgba(37, 99, 235, 0.12);}
.feature-icon {
width: 52px; height: 52px; border-radius: 15px; display: flex;
align-items: center; justify-content: center; font-size: 1.5rem; margin-bottom: 0.9rem;
}
.feature-title {font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.12rem;}
.feature-text {color: var(--muted); font-size: 0.92rem; line-height: 1.6; margin-top: 0.35rem;}

/* Steps timeline */
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; position: relative;}
.step {
background: #FFFFFF; border: 1px dashed #B9C9E6; border-radius: 16px;
padding: 1.1rem; position: relative;
}
.step-num {
width: 34px; height: 34px; border-radius: 50%; background: var(--navy); color: #FFFFFF;
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700;
display: flex; align-items: center; justify-content: center; margin-bottom: 0.7rem;
}
.step-title {font-weight: 600; font-size: 0.98rem;}
.step-text {color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; line-height: 1.5;}

/* Audience chips */
.chip {
display: inline-flex; align-items: center; gap: 0.4rem;
padding: 0.5rem 0.95rem; margin: 0.3rem 0.4rem 0.3rem 0;
background: #FFFFFF; border: 1px solid var(--line); border-radius: 999px;
font-size: 0.9rem; font-weight: 500;
}

/* Call to action band */
.cta {
background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 70%, #3B82F6 100%);
border-radius: 24px; padding: 2rem 2.2rem; color: #FFFFFF;
position: relative; overflow: hidden;
}
.cta::after {
content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
border-radius: 50%; background: rgba(250, 204, 21, 0.16);
}
.cta h3, .cta p {position: relative; z-index: 1;}
.cta h3 {color: #FFFFFF !important; font-size: 1.6rem; margin: 0;}
.cta p {color: #DBE6FF !important; margin: 0.4rem 0 0 0;}

/* Buttons */
.stButton > button {
border-radius: 12px; font-weight: 600; padding: 0.65rem 1.3rem;
border: 1px solid var(--line); background: #FFFFFF; color: var(--ink);
transition: all 0.15s ease;
}
.stButton > button:hover {border-color: var(--blue); color: var(--blue); transform: translateY(-1px);}
.stButton > button[kind="primary"] {
background: var(--yellow); color: var(--ink); border: none;
box-shadow: 0 8px 20px rgba(250, 204, 21, 0.4); font-size: 1.05rem;
}
.stButton > button[kind="primary"]:hover {background: #FDE047; color: var(--ink);}
.stButton > button p {font-weight: 600;}

.footer-note {font-size: 0.8rem !important; color: var(--muted) !important; text-align: center; margin-top: 2.4rem;}

/* Chat page */
.chat-hero {
background: #FFFFFF; border: 1px solid var(--line); border-radius: 20px;
padding: 1.4rem 1.6rem; box-shadow: 0 10px 30px rgba(15, 30, 74, 0.06);
}
.chat-hero h2 {margin: 0; font-size: 1.6rem;}
.chat-hero p {color: var(--muted) !important; margin: 0.3rem 0 0 0;}

[data-testid="stChatMessage"] {
background: #FFFFFF; border: 1px solid var(--line); border-radius: 18px;
padding: 1rem 1.1rem; box-shadow: 0 4px 14px rgba(15, 30, 74, 0.05);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
background: #EEF4FF; border-color: #C9DAFB;
}
[data-testid="stChatMessage"] table {border-collapse: collapse; width: 100%;}
[data-testid="stChatMessage"] th {
background: #EEF4FF; text-align: left; padding: 0.45rem; border-bottom: 2px solid var(--blue);
}
[data-testid="stChatMessage"] td {border-bottom: 1px solid var(--line); padding: 0.45rem;}

[data-testid="stBottom"] > div {background: transparent;}
[data-testid="stChatInput"] {
border: 2px solid var(--blue) !important; border-radius: 16px; background: #FFFFFF;
box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
}

.source-tag {
display: inline-block; font-size: 0.75rem; font-weight: 600; color: var(--navy);
background: #FEF9C3; border-radius: 6px; padding: 0.15rem 0.55rem; margin-top: 0.5rem;
}

/* Sidebar */
[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid var(--line);}
.side-std {
display: flex; align-items: center; gap: 0.6rem; background: #F5F8FE;
border-radius: 12px; padding: 0.6rem 0.75rem; margin: 0.45rem 0; font-size: 0.88rem;
}
.side-std b {font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.1rem; color: var(--blue);}

.ergit-error {
border: 1px solid #FBC4BE; background: #FEF1EF; color: #9F2D22;
border-radius: 14px; padding: 1rem 1.1rem;
}

@media (max-width: 760px) {
.steps {grid-template-columns: 1fr 1fr;}
.section-title {font-size: 1.5rem;}
.block-container::before {display: none;}
}
</style>
""")


def topbar():
    html(f"""
<div class="topbar">
<div class="pill"><span class="dot"></span> Lagos State · 20 LGAs · Public schools</div>
<div class="brand">
<div>
<div class="brand-name">ERGIT</div>
<div class="brand-sub">Education Resource-Gap Intelligence Tool</div>
</div>
{LOGO_SVG}
</div>
</div>
""")


inject_css()


# --------------------------------------------------
# 3. Load environment variables
# --------------------------------------------------

load_dotenv()


def get_key(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name)


api_key = get_key("OPENAI_API_KEY")
embedding_api_key = get_key("EMBEDDING_API_KEY")


# --------------------------------------------------
# 4. Initialize models and Chroma database
# --------------------------------------------------

@st.cache_resource
def load_rag_components():
    embeddings = OpenAIEmbeddings(
        model="Qwen/Qwen3-Embedding-0.6B",
        base_url="https://qwen-embed.publicaai.com/v1",
        api_key=embedding_api_key
    )

    vector_db = Chroma(
        collection_name="lerkb",
        embedding_function=embeddings,
        persist_directory=os.path.join(BASE_DIR, "chroma_store")
    )

    chat_model = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-20b",
        temperature=0
    )

    return vector_db, chat_model


# --------------------------------------------------
# 5. Create the prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are the Lagos Education Resource Assistant.

Answer the user's question using only the supplied context.

Instructions:
- Be clear, helpful, and concise.
- Use tables when they make comparisons easier to understand.
- Preserve LGA names, figures, rankings, and standards accurately.
- Resource-gap scores are relative scores out of 100, not percentages
  of missing resources.
- Do not invent facts, statistics, rankings, or citations.
- If the context does not contain the answer, say so clearly.
- Distinguish between a resource-gap score and an official national
  education standard.
- Do not claim that a source supports information it does not contain.

Context:
{context}

Question:
{question}

Answer:
""")


# --------------------------------------------------
# 6. Route questions to the appropriate source
# --------------------------------------------------

def retrieve_context(question, vector_db, general_retriever):
    ranking_keywords = [
        "ranking",
        "ranked",
        "greatest overall",
        "greatest resource gap",
        "greatest resource gaps",
        "largest resource gap",
        "largest resource gaps",
        "highest resource gap",
        "highest resource gaps",
        "highest resource-gap score",
        "lowest resource gap",
        "lowest resource gaps",
        "smallest resource gap",
        "smallest resource gaps",
        "top five",
        "bottom five",
        "five lgas",
        "which lgas",
        "overall resource gap",
        "overall resource gaps",
        "most lacking overall",
        "worst resource gaps",
        "resource-gap score",
        "resource gap scores"
    ]

    question_lower = question.lower()

    is_ranking_question = any(
        keyword in question_lower
        for keyword in ranking_keywords
    )

    if is_ranking_question:
        docs = vector_db.similarity_search(
            "LAGOS STATE EDUCATION RESOURCE-GAP RANKING",
            k=2,
            filter={
                "source": "Lagos_Resource_Gap_Ranking.txt"
            }
        )

        docs.sort(
            key=lambda doc: (
                0 if "LAGOS STATE EDUCATION RESOURCE-GAP RANKING"
                in doc.page_content.upper() else 1
            )
        )

        source_used = "Ranking TXT"

    else:
        docs = general_retriever.invoke(question)
        source_used = "Education PDF"

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return context, source_used


# --------------------------------------------------
# 7. Answer questions
# --------------------------------------------------

def answer_question(question, vector_db, chat_model, general_retriever):
    context, source_used = retrieve_context(
        question, vector_db, general_retriever
    )

    if not context.strip():
        return (
            "I could not find relevant information in the "
            "available education documents.",
            source_used
        )

    chain = prompt | chat_model | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return answer, source_used


# --------------------------------------------------
# 8. 3D hero (floating books, pencils, caps and data bars)
# --------------------------------------------------

HERO_3D = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@600;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
html, body {margin: 0; height: 100%; overflow: hidden; font-family: 'Inter', sans-serif;}
.wrap {
position: relative; height: 100%; border-radius: 28px; overflow: hidden;
background:
radial-gradient(circle at 78% 30%, rgba(250,204,21,0.28), transparent 40%),
radial-gradient(circle at 92% 85%, rgba(16,185,129,0.22), transparent 38%),
linear-gradient(135deg, #0F1E4A 0%, #1E3A8A 55%, #2563EB 100%);
}
.grid {
position: absolute; inset: 0;
background-image:
linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px);
background-size: 40px 40px;
}
canvas {position: absolute; inset: 0;}
.text {
position: absolute; left: 6%; top: 50%; transform: translateY(-50%);
max-width: 520px; color: #FFFFFF; z-index: 2;
}
.badge {
display: inline-block; background: rgba(255,255,255,0.12);
border: 1px solid rgba(255,255,255,0.25); border-radius: 999px;
padding: 6px 14px; font-size: 13px; letter-spacing: 0.04em; color: #E0E9FF;
backdrop-filter: blur(6px);
}
h1 {
font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800;
font-size: 46px; line-height: 1.08; margin: 18px 0 14px 0; letter-spacing: -0.01em;
}
h1 span {color: #FACC15;}
p {font-size: 17px; line-height: 1.6; color: #D6E2FF; margin: 0;}
.row {display: flex; gap: 22px; margin-top: 26px;}
.mini b {display: block; font-family: 'Plus Jakarta Sans', 'Inter', sans-serif; font-size: 26px; color: #FFFFFF;}
.mini span {font-size: 12.5px; color: #B8C8F0;}
@media (max-width: 760px) {
h1 {font-size: 32px;} .text {left: 7%; right: 7%;} canvas {opacity: 0.35;}
}
</style>
</head>
<body>
<div class="wrap">
<div class="grid"></div>
<canvas id="scene"></canvas>
<div class="text">
<div class="badge">🎓 Education Resource-Gap Intelligence Tool</div>
<h1>Every child deserves a <span>teacher</span>, a classroom and a clean toilet.</h1>
<p>ERGIT shows which Lagos LGAs are furthest from the national school standards, and where support should go first.</p>
<div class="row">
<div class="mini"><b>20</b><span>LGAs analysed</span></div>
<div class="mini"><b>360</b><span>data records</span></div>
<div class="mini"><b>3</b><span>UBE standards</span></div>
</div>
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function () {
if (!window.THREE) return;
const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
camera.position.set(0, 0.6, 12);

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(5, 8, 6); scene.add(sun);
const warm = new THREE.PointLight(0xfacc15, 0.8, 30);
warm.position.set(6, 2, 5); scene.add(warm);

const group = new THREE.Group();
scene.add(group);
const mat = c => new THREE.MeshStandardMaterial({color: c, roughness: 0.45, metalness: 0.08});
const pages = mat(0xFFFDF5);
const floaters = [];

function book(color, w, h, d) {
  const g = new THREE.Group();
  const cover = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(color));
  const pg = new THREE.Mesh(new THREE.BoxGeometry(w * 0.94, h * 0.78, d * 1.01), pages);
  pg.position.x = 0.03;
  g.add(cover); g.add(pg);
  return g;
}

function pencil(color) {
  const g = new THREE.Group();
  const body = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 2.4, 6), mat(color));
  const wood = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.36, 6), mat(0xF2C9A0));
  wood.position.y = -1.38;
  const lead = new THREE.Mesh(new THREE.ConeGeometry(0.045, 0.13, 6), mat(0x1F2937));
  lead.position.y = -1.6;
  const band = new THREE.Mesh(new THREE.CylinderGeometry(0.125, 0.125, 0.16, 12), mat(0xC0C7D6));
  band.position.y = 1.2;
  const eraser = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.2, 12), mat(0xF97066));
  eraser.position.y = 1.38;
  g.add(body, wood, lead, band, eraser);
  return g;
}

function cap() {
  const g = new THREE.Group();
  const board = new THREE.Mesh(new THREE.BoxGeometry(1.7, 0.08, 1.7), mat(0x111827));
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.62, 0.45, 24), mat(0x111827));
  base.position.y = -0.27;
  const button = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.05, 12), mat(0xFACC15));
  button.position.y = 0.06;
  const cord = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.85), mat(0xFACC15));
  cord.position.set(0.42, 0.06, 0.42); cord.rotation.y = Math.PI / 4;
  const tassel = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.08, 0.5, 8), mat(0xFACC15));
  tassel.position.set(0.82, -0.2, 0.82);
  g.add(board, base, button, cord, tassel);
  return g;
}

// Stack of books (right side)
const stack = new THREE.Group();
const s1 = book(0x2563EB, 2.6, 0.42, 1.8); s1.position.y = 0;
const s2 = book(0xFACC15, 2.3, 0.38, 1.6); s2.position.y = 0.42; s2.rotation.y = 0.25;
const s3 = book(0x10B981, 2.0, 0.34, 1.45); s3.position.y = 0.8; s3.rotation.y = -0.2;
stack.add(s1, s2, s3);
stack.position.set(4.2, -1.9, 0); stack.rotation.set(0.25, -0.5, 0);
group.add(stack);

// Rising data bars standing on the stack
const bars = [];
const barColors = [0x93C5FD, 0x60A5FA, 0x3B82F6, 0xFACC15];
for (let i = 0; i < 4; i++) {
  const b = new THREE.Mesh(new THREE.BoxGeometry(0.32, 1, 0.32), mat(barColors[i]));
  b.position.set(-0.6 + i * 0.42, 1.0, 0);
  bars.push(b); stack.add(b);
}

// Floating objects
const cap1 = cap(); cap1.position.set(6.6, 2.4, -1); group.add(cap1);
floaters.push({o: cap1, base: 2.4, speed: 0.9, spin: 0.35});

const p1 = pencil(0xFACC15); p1.position.set(2.0, 1.7, 1); p1.rotation.set(0.3, 0, -0.9); group.add(p1);
floaters.push({o: p1, base: 1.7, speed: 1.2, spin: 0.0});

const p2 = pencil(0x2563EB); p2.position.set(7.4, -0.6, 0.5); p2.rotation.set(-0.4, 0, 0.6); group.add(p2);
floaters.push({o: p2, base: -0.6, speed: 1.0, spin: 0.0});

const fb = book(0xF97066, 1.4, 0.26, 1.0); fb.position.set(2.4, -0.4, 2); fb.rotation.set(0.6, 0.4, 0.3); group.add(fb);
floaters.push({o: fb, base: -0.4, speed: 0.8, spin: 0.4});

const fb2 = book(0x8B5CF6, 1.2, 0.22, 0.9); fb2.position.set(5.9, 0.2, 2.2); fb2.rotation.set(-0.3, 0.9, -0.2); group.add(fb2);
floaters.push({o: fb2, base: 0.2, speed: 1.1, spin: -0.3});

// Small glowing particles (ideas)
const pts = new THREE.BufferGeometry();
const n = 140, arr = new Float32Array(n * 3);
for (let i = 0; i < n; i++) {
  arr[i * 3] = (Math.random() - 0.2) * 16;
  arr[i * 3 + 1] = (Math.random() - 0.5) * 8;
  arr[i * 3 + 2] = (Math.random() - 0.5) * 6;
}
pts.setAttribute('position', new THREE.BufferAttribute(arr, 3));
const dots = new THREE.Points(pts, new THREE.PointsMaterial({color: 0xFFFFFF, size: 0.05, transparent: true, opacity: 0.6}));
scene.add(dots);

let mx = 0, my = 0;
window.addEventListener('mousemove', e => {
  mx = (e.clientX / window.innerWidth - 0.5);
  my = (e.clientY / window.innerHeight - 0.5);
});

function resize() {
  const w = canvas.parentElement.clientWidth, h = canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.position.x = w < 760 ? 3.2 : 0;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

const clock = new THREE.Clock();
function animate() {
  const t = clock.getElapsedTime();
  floaters.forEach((f, i) => {
    f.o.position.y = f.base + Math.sin(t * f.speed + i) * 0.25;
    f.o.rotation.y += f.spin * 0.01;
  });
  bars.forEach((b, i) => {
    const h = 0.9 + i * 0.45 + Math.sin(t * 1.5 + i) * 0.18;
    b.scale.y = h; b.position.y = 0.95 + h / 2 - 0.5;
  });
  stack.rotation.y = -0.5 + Math.sin(t * 0.3) * 0.12;
  dots.rotation.y = t * 0.02;
  group.rotation.y += ((mx * 0.25) - group.rotation.y) * 0.05;
  group.rotation.x += ((my * 0.12) - group.rotation.x) * 0.05;
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------
# 9. Sidebar
# --------------------------------------------------

def render_sidebar():
    with st.sidebar:
        html(f"""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.8rem;">
{LOGO_SVG}
<div>
<div class="brand-name" style="font-size:1.25rem;">ERGIT</div>
<div class="brand-sub">Lagos State, Nigeria</div>
</div>
</div>
<p style="font-size:0.88rem; color:#5B6680; line-height:1.55;">
A data tool that ranks Lagos LGAs by how urgently their public schools
need teachers, classrooms and toilets.
</p>
<div class="eyebrow" style="margin-top:1.2rem;">UBE minimum standards</div>
<div class="side-std">👩🏾‍🏫 <span><b>35</b> students per teacher</span></div>
<div class="side-std">🏫 <span><b>35</b> students per classroom</span></div>
<div class="side-std">🚻 <span><b>30</b> students per toilet</span></div>
<p style="font-size:0.8rem; color:#5B6680; margin-top:1.2rem; line-height:1.5;">
Answers come only from the project's documents. If the data does not
cover a question, ERGIT will say so.
</p>
""")


# --------------------------------------------------
# 10. Page 1: Home (what ERGIT does)
# --------------------------------------------------

def render_home():
    topbar()

    components.html(HERO_3D, height=470)

    html('<div class="gap"></div>')

    # Key findings
    html("""
<div class="eyebrow">Key findings · 2024/25</div>
<div class="section-title">Facilities are the <span class="hl">biggest gap</span> in Lagos public schools</div>
<p class="lead">Each LGA was compared with the Universal Basic Education (UBE) standards.
Here is how many of the 20 LGAs fall short.</p>
<div style="height:1rem;"></div>
""")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        html("""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:#FEE2E2;">🏫</div></div>
<div class="stat-num">20<small>/20</small></div>
<div class="stat-label">LGAs lack classrooms</div>
<div class="bar"><span style="width:100%; background:#F97066;"></span></div>
<div class="stat-note">Standard: 35 per classroom</div>
</div>
""")
    with c2:
        html("""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:#FEF3C7;">🚻</div></div>
<div class="stat-num">19<small>/20</small></div>
<div class="stat-label">LGAs lack toilets</div>
<div class="bar"><span style="width:95%; background:#F59E0B;"></span></div>
<div class="stat-note">Standard: 30 per toilet</div>
</div>
""")
    with c3:
        html("""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:#DBEAFE;">👩🏾‍🏫</div></div>
<div class="stat-num">9<small>/20</small></div>
<div class="stat-label">LGAs lack teachers</div>
<div class="bar"><span style="width:45%; background:#2563EB;"></span></div>
<div class="stat-note">Standard: 35 per teacher</div>
</div>
""")
    with c4:
        html("""
<div class="stat">
<div class="stat-top"><div class="stat-icon" style="background:#D1FAE5;">🎒</div></div>
<div class="stat-num">JSS</div>
<div class="stat-label">Most pressured level</div>
<div class="bar"><span style="width:100%; background:#10B981;"></span></div>
<div class="stat-note">Junior secondary schools</div>
</div>
""")

    html("""
<p style="font-size:0.8rem; color:#5B6680; margin-top:0.8rem;">
Public Primary, JSS and SSS schools. Sources: DNEMIS Annual School Census 2024/25; Lagos Bureau of Statistics.
</p>
<div class="gap"></div>
""")

    # What ERGIT does
    html("""
<div class="eyebrow">What ERGIT does</div>
<div class="section-title">From scattered reports to <span class="hl">clear answers</span></div>
<p class="lead">Nigeria's education data sits in many separate reports. ERGIT brings it together,
so decision-makers can see where support should go first.</p>
<div style="height:1rem;"></div>
""")

    f1, f2, f3 = st.columns(3)
    with f1:
        html("""
<div class="feature">
<div class="feature-icon" style="background:#DBEAFE;">📊</div>
<div class="feature-title">Ranks every LGA</div>
<div class="feature-text">Compares all 20 Lagos LGAs with the national standards for teachers, classrooms and toilets, and ranks them by need.</div>
</div>
""")
    with f2:
        html("""
<div class="feature">
<div class="feature-icon" style="background:#FEF3C7;">📈</div>
<div class="feature-title">Forecasts shortages</div>
<div class="feature-text">Machine learning learns from past years to warn where teacher shortages are likely to grow.</div>
</div>
""")
    with f3:
        html("""
<div class="feature">
<div class="feature-icon" style="background:#D1FAE5;">💬</div>
<div class="feature-title">Answers your questions</div>
<div class="feature-text">An AI assistant explains the results in plain English, using only verified data, and shows its source.</div>
</div>
""")

    html('<div class="gap"></div>')

    # How it works
    html("""
<div class="eyebrow">How it works</div>
<div class="section-title">Four steps, <span class="hl">one clear picture</span></div>
<div style="height:1rem;"></div>
<div class="steps">
<div class="step"><div class="step-num">1</div><div class="step-title">📥 Collect</div>
<div class="step-text">Official data from DNEMIS and the Lagos Bureau of Statistics.</div></div>
<div class="step"><div class="step-num">2</div><div class="step-title">🧹 Clean &amp; compare</div>
<div class="step-text">Checked, cleaned and measured against the UBE standards.</div></div>
<div class="step"><div class="step-num">3</div><div class="step-title">🏆 Rank &amp; forecast</div>
<div class="step-text">LGAs ranked by need; shortages forecast with machine learning.</div></div>
<div class="step"><div class="step-num">4</div><div class="step-title">💡 Explain</div>
<div class="step-text">The AI assistant answers questions with sources.</div></div>
</div>
<div class="gap"></div>
""")

    # Who it is for
    html("""
<div class="eyebrow">Who it is for</div>
<div class="section-title">Built for people who <span class="hl">decide and fund</span></div>
<div style="height:0.6rem;"></div>
<div>
<span class="chip">🏛️ Lagos SUBEB</span>
<span class="chip">📚 Ministry of Basic &amp; Secondary Education</span>
<span class="chip">🇳🇬 UBEC</span>
<span class="chip">🤝 NGOs</span>
<span class="chip">💼 Donors &amp; CSR teams</span>
<span class="chip">🔬 Researchers</span>
</div>
<div class="gap"></div>
""")

    # Call to action
    left, right = st.columns([3, 1.3])
    with left:
        html("""
<div class="cta">
<h3>Ready to explore the data?</h3>
<p>Ask ERGIT which LGAs need teachers, classrooms or toilets most.</p>
</div>
""")
    with right:
        html('<div style="height:3.4rem;"></div>')
        if st.button("Start asking ERGIT  →", type="primary", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    html("""
<p class="footer-note">
ERGIT is a student data science capstone project. Figures come from official public sources.
</p>
""")


# --------------------------------------------------
# 11. Page 2: Chat with ERGIT
# --------------------------------------------------

SAMPLE_QUESTIONS = [
    ("🏆", "Which LGAs have the greatest overall resource gaps?"),
    ("👩🏾‍🏫", "Which LGAs need teachers most?"),
    ("🚻", "How does Badagry compare with the toilet standard?"),
    ("🎒", "Which school level is under the most pressure?")
]


def render_chat():
    topbar()

    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        if st.button("←  Back to home"):
            st.session_state.page = "home"
            st.rerun()
    with nav_right:
        if st.session_state.messages:
            if st.button("🧹 Clear chat", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    html("""
<div class="chat-hero">
<div class="eyebrow">ERGIT Assistant</div>
<h2 class="jakarta">Ask about Lagos schools 🎓</h2>
<p>Teachers, classrooms, toilets and resource gaps across the 20 Lagos LGAs.</p>
</div>
<div style="height:1rem;"></div>
""")

    if not api_key or not embedding_api_key:
        html("""
<div class="ergit-error">
<b>API keys are missing.</b><br>
On Streamlit Cloud, add OPENAI_API_KEY and EMBEDDING_API_KEY under
Manage app → Settings → Secrets. When running locally, add them to your .env file.
</div>
""")
        st.stop()

    try:
        vector_db, chat_model = load_rag_components()
    except Exception as e:
        html(f'<div class="ergit-error"><b>Could not initialize the RAG system:</b> {e}</div>')
        st.stop()

    general_retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 5,
            "filter": {
                "source": "Lagos_Education_RAG_Knowledge_Base.pdf"
            }
        }
    )

    if not st.session_state.messages:
        html('<p class="eyebrow" style="margin-top:0.4rem;">Try a question</p>')
        q1, q2 = st.columns(2)
        for i, (icon, sample) in enumerate(SAMPLE_QUESTIONS):
            column = q1 if i % 2 == 0 else q2
            with column:
                if st.button(f"{icon}  {sample}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_question = sample
                    st.rerun()

    for message in st.session_state.messages:
        avatar = "🧑🏾" if message["role"] == "user" else "🎓"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "source" in message:
                html(f'<span class="source-tag">📄 Source used: {message["source"]}</span>')

    typed_question = st.chat_input(
        "Ask about education resources in Lagos State..."
    )

    question = typed_question or st.session_state.pop("pending_question", None)

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🧑🏾"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Searching the education documents..."):
                try:
                    answer, source_used = answer_question(
                        question, vector_db, chat_model, general_retriever
                    )

                    st.markdown(answer)
                    html(f'<span class="source-tag">📄 Source used: {source_used}</span>')

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source": source_used
                    })

                except Exception as e:
                    st.error(
                        "Something went wrong while answering your question."
                    )
                    st.code(f"{type(e).__name__}: {e}")
                    print(f"RAG error: {type(e).__name__}: {e}")


# --------------------------------------------------
# 12. Run the app
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "messages" not in st.session_state:
    st.session_state.messages = []

render_sidebar()

if st.session_state.page == "home":
    render_home()
else:
    render_chat()