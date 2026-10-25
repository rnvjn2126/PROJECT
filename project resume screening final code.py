import sys
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- Import torch/sentence-transformers BEFORE tensorflow (avoids DLL/protobuf clashes) ----
ADV_ERR = ""
try:
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    ADV_OK = True
except Exception as e:
    ADV_OK = False
    ADV_ERR = f"{type(e).__name__}: {e}"
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

import streamlit as st
import numpy as np
import pandas as pd
from datasets import load_dataset
import pdfplumber
import re
import io
import warnings
import requests
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go


from langdetect import detect, DetectorFactory, LangDetectException
from deep_translator import GoogleTranslator
DetectorFactory.seed = 0


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


warnings.filterwarnings("ignore")



class CoverLetterGenerator:
    def __init__(self):
        self.templates = {
            "TECH": {
                "hook": "As a passionate {level} {skills_summary} professional, I was excited to discover your {job_title} position at {company}.",
                "skills": "My hands-on expertise with {top_skills} and proven ability to deliver {achievements} perfectly align with your technical requirements.",
                "company": "I am particularly impressed by {company}'s leadership in {company_focus} and commitment to innovation.",
                "cta": "I am eager to bring my technical expertise to {company} and contribute to your cutting-edge projects."
            },
            "MANAGEMENT": {
                "hook": "With {years} years of leadership experience driving {domain} initiatives, your {job_title} role at {company} immediately caught my attention.",
                "skills": "My track record of {achievements} and expertise in team leadership directly supports your organizational goals.",
                "company": "{company}'s reputation for {company_focus} excellence resonates deeply with my professional values.",
                "cta": "I look forward to leveraging my strategic leadership to drive {company}'s continued success."
            },
            "FINANCE": {
                "hook": "As a results-driven finance professional with expertise in {skills_summary}, I am enthusiastic about your {job_title} opportunity at {company}.",
                "skills": "My experience optimizing {achievements} positions me to deliver immediate value to your financial operations.",
                "company": "{company}'s strategic focus on {company_focus} aligns perfectly with my career objectives.",
                "cta": "I would welcome the opportunity to discuss how my financial expertise can support {company}'s growth."
            },
            "GENERAL": {
                "hook": "Your {job_title} position at {company} represents an exciting opportunity to leverage your {skills_summary} background.",
                "skills": "I offer proven success in {top_skills} and a demonstrated ability to {key_achievement}.",
                "company": "{company}'s commitment to {company_focus} excellence strongly aligns with my professional aspirations.",
                "cta": "I am confident my skills and experience would make valuable contributions to your team."
            }
        }


    def generate(self, job_title, company, job_desc, resume_text, skills, level, category="GENERAL"):
        text_lower = resume_text.lower()
        achievements = []
        for line in resume_text.split("\n"):
            line = line.strip()
            if len(line) > 20 and any(word in line.lower() for word in ['increased', 'reduced', 'improved', 'achieved', 'delivered', 'grew']):
                achievements.append(line[:100])


        years = "2+" if any(str(y) in text_lower for y in range(18, 25)) else "5+"
        top_skills_str = ", ".join(skills[:5]) if skills else "core competencies"


        job_reqs = []
        key_phrases = ['requirements:', 'must have', 'qualifications', 'looking for', 'responsibilities']
        for phrase in key_phrases:
            if phrase.lower() in job_desc.lower():
                start = job_desc.lower().index(phrase.lower())
                snippet = job_desc[max(0, start - 50):start + 150]
                job_reqs.append(snippet[:60].strip())


        company_focus = "technological innovation" if any(word in company.lower() for word in ['tech', 'software', 'digital']) else "operational excellence"
        key_achievement = achievements[0][:50] + "..." if achievements else "drive results"
        template = self.templates.get(category, self.templates["GENERAL"])


        hook = template["hook"].format(level=level.lower(), skills_summary=top_skills_str, job_title=job_title, company=company, domain="strategic", years=years)
        skills_line = template["skills"].format(top_skills=top_skills_str, achievements=" and ".join([a[:30] + "..." for a in achievements[:2]]) if achievements else "measurable results", job_reqs=", ".join(job_reqs) if job_reqs else "your outlined requirements", key_achievement=key_achievement)
        company_line = template["company"].format(company=company, company_focus=company_focus, industry="technology" if "tech" in category.upper() else "your industry")
        cta_line = template["cta"].format(company=company)


        return (
            "Dear Hiring Manager,\n\n" + hook + "\n\n" + skills_line + "\n\n" + company_line + "\n\n" + cta_line + "\n\n"
            + "Best regards,\n[Your Name]\n[Your Phone] | [Your Email] | [LinkedIn Profile]"
        ).strip()



cover_gen = CoverLetterGenerator()



# =========================================================
# LANGUAGE NORMALIZATION ENGINE
# =========================================================
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "tr": "Turkish", "pl": "Polish", "uk": "Ukrainian", "sv": "Swedish",
    "fi": "Finnish", "no": "Norwegian", "he": "Hebrew", "el": "Greek", "cs": "Czech", "ro": "Romanian",
    "hu": "Hungarian", "fa": "Persian"
}


TRANSLATOR_LANG_MAP = {"zh-cn": "zh-CN", "zh-tw": "zh-TW"}



def detect_resume_language(text):
    sample = text.strip()
    if len(sample) < 20:
        return "en", "English"
    try:
        code = detect(sample[:3000])
    except LangDetectException:
        return "en", "English"
    return code, LANG_NAMES.get(code, code.upper())



def translate_to_english(text, source_lang, chunk_size=4500):
    if source_lang == "en":
        return text


    translator_code = TRANSLATOR_LANG_MAP.get(source_lang, source_lang)
    chunks, current = [], ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > chunk_size:
            if current:
                chunks.append(current)
            current = line
        else:
            current = current + "\n" + line if current else line
    if current:
        chunks.append(current)


    translator = GoogleTranslator(source=translator_code, target="en")
    translated_parts = []
    for chunk in chunks:
        try:
            translated_parts.append(translator.translate(chunk) if chunk.strip() else chunk)
        except Exception:
            translated_parts.append(chunk)
    return "\n".join(translated_parts)



def normalize_resume_to_english(raw_text):
    lang_code, lang_name = detect_resume_language(raw_text)
    english_text = raw_text if lang_code == "en" else translate_to_english(raw_text, lang_code)
    return {
        "original_text": raw_text,
        "english_text": english_text,
        "detected_lang_code": lang_code,
        "detected_lang_name": lang_name,
        "was_translated": lang_code != "en"
    }



# =========================================================
# UI SETUP
# =========================================================
st.set_page_config(page_title="CareerAI Pro", page_icon="⚡", layout="wide")


st.markdown("""
<style>
.stApp { background: radial-gradient(circle at top left, #1e293b 0, #020617 45%), radial-gradient(circle at bottom right, #0f172a 0, #020617 55%); color: #e2e8f0; }
h1 { background: linear-gradient(90deg,#38bdf8,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
div[data-testid="stMetric"] { background: rgba(15,23,42,0.85); border: 1px solid rgba(148,163,184,0.5); border-radius: 14px; padding: 10px; }
div.stButton > button { background: linear-gradient(90deg,#4f46e5,#3b82f6); border-radius: 999px; font-weight: 600; }
.kpi-card { background: rgba(15,23,42,0.95); border: 1px solid rgba(59,130,246,0.3); border-radius: 20px; padding: 18px; }
.heatmap-high { background-color: rgba(74, 222, 128, 0.2); padding: 5px; border-radius: 4px; border-left: 4px solid #4ade80; margin-bottom: 6px; }
.heatmap-low { opacity: 0.6; padding: 2px 5px; margin-bottom: 2px; font-size: 0.9em; }
.cover-letter { background-color: #f8fafc; color: #0f172a; padding: 25px; border-radius: 12px; font-family: Georgia, serif; line-height: 1.7; border-left: 5px solid #3b82f6; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.lang-badge { background: rgba(56,189,248,0.15); border: 1px solid rgba(56,189,248,0.5); color: #38bdf8; padding: 8px 14px; border-radius: 999px; font-weight: 600; display: inline-block; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=70)
with col2:
    st.title("🚀 CareerAI Pro: Ultimate Edition")
    st.caption("⚡ TensorFlow NN • Scikit-Learn Bias Audit • Local Cover Letters • 🌍 Multilingual Resume Support")


st.markdown("---")



# =========================================================
# DATA ENGINE
# =========================================================
@st.cache_data(show_spinner=True)
def load_data():
    st.info("🔄 Loading LinkedIn job data...")
    try:
        ds = load_dataset("datastax/linkedin_job_listings", split="train[:1200]")
        df = ds.to_pandas().rename(columns={"title": "job_title", "description": "job_description"})
        df = df.dropna(subset=["job_title", "job_description"]).reset_index(drop=True)


        def extract_meta(row):
            text = (row['job_description'] + " " + row['job_title']).lower()
            company = "Unknown"
            if ' at ' in row['job_title'].lower():
                parts = row['job_title'].split(' at ')
                if len(parts) > 1:
                    company = parts[-1].strip()


            matches = re.findall(r'\$(\d{2,3}(?:,\d{3})*|\d{2,3}k)', text)
            values = []
            for value in matches:
                try:
                    number = int(value.replace(',', '').replace('k', '000'))
                    if 25000 <= number <= 400000:
                        values.append(number)
                except ValueError:
                    pass
            salary = np.mean(values) if values else None
            return pd.Series([company, salary])


        df[['company_name', 'est_salary']] = df.apply(extract_meta, axis=1)


        def categorize(title):
            title = str(title).lower()
            if any(x in title for x in ['attorney', 'lawyer', 'legal']): return 'LEGAL'
            if any(x in title for x in ['nurse', 'doctor', 'medical', 'health']): return 'HEALTHCARE'
            if any(x in title for x in ['finance', 'accountant', 'banking']): return 'FINANCE'
            if any(x in title for x in ['engineer', 'developer', 'python', 'software']): return 'TECH'
            if any(x in title for x in ['manager', 'director', 'executive']): return 'MANAGEMENT'
            return 'GENERAL'


        df["category"] = df["job_title"].apply(categorize)
        df["job_doc"] = df["job_title"] + " " + df["category"] + " " + df["job_description"]
        vect = TfidfVectorizer(max_features=2000, stop_words="english")
        X = vect.fit_transform(df["job_doc"].fillna(""))
        return df, vect, X
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame(), None, None



df_jobs, tfidf_vect, X_jobs = load_data()
ALL_CATEGORIES = sorted(df_jobs["category"].unique().tolist()) if not df_jobs.empty else []



# =========================================================
# MODELS
# =========================================================
@st.cache_resource
def train_rf_model(_df, _X_matrix):
    if _df.empty:
        return None
    train_df = _df.dropna(subset=['est_salary'])
    if len(train_df) < 10:
        return None
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(_X_matrix[train_df.index], train_df['est_salary'])
    return model



rf_model = train_rf_model(df_jobs, X_jobs)



def train_tf_model(_df, _X_matrix):
    tf.keras.backend.clear_session()
    train_df = _df.dropna(subset=['est_salary'])
    if len(train_df) < 10:
        return None, None, None


    X_train = _X_matrix[train_df.index].toarray()
    y_train = train_df['est_salary'].values
    y_mean, y_std = y_train.mean(), y_train.std()
    if y_std == 0:
        y_std = 1
    y_scaled = (y_train - y_mean) / y_std


    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_scaled, epochs=15, batch_size=32, verbose=0, validation_split=0.2)
    return model, history, (y_mean, y_std)



def audit_resume_bias(resume_text):
    strong_words = ['managed', 'led', 'developed', 'created', 'engineered', 'launched', 'solved', 'architected', 'mitigated']
    weak_words = ['helped', 'assisted', 'supported', 'tried', 'learned', 'responsible for', 'participated']
    cv = CountVectorizer(vocabulary=strong_words + weak_words)
    try:
        counts = cv.fit_transform([resume_text]).toarray()[0]
        strong_count = sum(counts[:len(strong_words)])
        weak_count = sum(counts[len(strong_words):])
        return strong_count, weak_count, "Strong / Active" if strong_count > weak_count else "Passive / Weak"
    except Exception:
        return 0, 0, "Neutral"



# =========================================================
# PDF / RESUME ANALYSIS
# =========================================================
def extract_pdf_text(pdf_file):
    pdf_file.seek(0)
    try:
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()
    except Exception:
        return ""



def is_valid_resume(text):
    text_lower = text.lower()

    has_email = bool(re.search(r'[\w.\-]+@[\w.\-]+\.\w+', text))
    has_phone = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text))

    sections = ['education', 'experience', 'skills', 'projects', 'summary', 'objective', 'certifications', 'work history']
    section_score = sum(section in text_lower for section in sections)
    return (has_email or has_phone) and section_score >= 2



def analyze_profile(text):
    text_lower = text.lower()
    tech = ['python', 'java', 'flutter', 'react', 'sql', 'pl/sql', 'lex', 'firebase', 'git', 'c++', 'html', 'css', 'javascript', 'aws', 'docker', 'gis', 'ndma', 'disaster management', 'emergency response']
    found = [skill.capitalize() if len(skill) > 3 else skill.upper() for skill in tech if skill in text_lower]
    student_words = ['student', 'university', 'bachelor', 'diploma', '2024', '2025', '2026', 'intern']
    level = "Entry-Level/Student" if any(word in text_lower for word in student_words) else "Professional"
    return list(set(found)), level



def score_jobs(resume_text, skills, level):
    resume_vec = tfidf_vect.transform([resume_text])
    cosine_sims = cosine_similarity(resume_vec, X_jobs)[0]
    df = df_jobs.copy()
    df['score'] = cosine_sims


    def calc_adj(row):
        penalty = 0.4 if level == "Entry-Level/Student" and any(
            word in row['job_title'].lower() for word in ['senior', 'manager', 'director', 'lead', 'principal']
        ) else 0.0
        skill_matches = sum(skill.lower() in str(row['job_description']).lower() for skill in skills)
        return min(skill_matches * 0.1, 0.5) - penalty


    df['final_score'] = df['score'] + df.apply(calc_adj, axis=1)
    max_score = df['final_score'].max()
    df['match_pct'] = 50.0 if max_score <= 0 else ((df['final_score'] / max_score) * 100).fillna(0).clip(0, 100)
    return df.sort_values('final_score', ascending=False)



# =========================================================
# ADVANCED ENGINE (additive): dense embeddings, trust fusion, XAI, ESCO
# =========================================================
EMBED_MODELS = {
    "Multilingual MiniLM (XLM-R family, 50+ langs)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "CareerBERT-JG (ESCO fine-tuned, German base)": "lwolfrum2/careerbert-jg",
    "CareerBERT-G (ESCO fine-tuned, German base)": "lwolfrum2/careerbert-g",
}

SECTION_ALIASES = {
    "experience": ["experience", "work experience", "employment history", "work history", "professional experience",
                   "expérience", "expérience professionnelle", "berufserfahrung", "experiencia", "experiencia laboral", "अनुभव", "அனுபவம்"],
    "projects": ["projects", "project", "projets", "projekte", "proyectos", "academic projects", "திட்டங்கள்"],
    "skills": ["skills", "technical skills", "key skills", "compétences", "kenntnisse", "fähigkeiten", "habilidades", "திறன்கள்"],
    "certifications": ["certifications", "certificates", "licenses", "zertifikate", "certificaciones"],
    "education": ["education", "formation", "ausbildung", "educación", "கல்வி"],
    "summary": ["summary", "objective", "profile", "about me", "profil", "resumen", "zusammenfassung"],
    "interests": ["interests", "hobbies", "hobbys", "intérêts", "loisirs", "intereses", "personal interests"],
}
SECTION_TRUST = {"experience": 1.0, "projects": 0.9, "certifications": 0.8, "skills": 0.8,
                 "education": 0.7, "summary": 0.6, "other": 0.5, "interests": 0.4}


def _classify_header(text):
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    for sec, aliases in SECTION_ALIASES.items():
        if t in aliases:
            return sec
    return None


def extract_layout_lines(pdf_file):
    pdf_file.seek(0)
    out = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            for pno, page in enumerate(pdf.pages):
                for ln in page.extract_text_lines(return_chars=True):
                    chars = ln.get("chars", [])
                    size = float(np.mean([c["size"] for c in chars])) if chars else 0
                    bold = bool(chars) and sum("bold" in c.get("fontname", "").lower() for c in chars) > len(chars) / 2
                    out.append({"text": ln["text"].strip(), "size": size, "bold": bold, "page": pno,
                                "bbox": (ln["x0"], ln["top"], ln["x1"], ln["bottom"])})
    except Exception:
        pass
    return out


def build_trusted_sections(pdf_file, norm):
    lines = extract_layout_lines(pdf_file)
    if not lines:
        lines = [{"text": l.strip(), "size": 0, "bold": False} for l in norm["original_text"].split("\n") if l.strip()]
    sections, cur, buf = [], "other", []
    for l in lines:
        sec = _classify_header(l["text"])
        if sec:
            if buf:
                sections.append((cur, "\n".join(buf)))
            cur, buf = sec, []
        else:
            buf.append(l["text"])
    if buf:
        sections.append((cur, "\n".join(buf)))
    result = []
    for sec, txt in sections:
        txt_en = translate_to_english(txt, norm["detected_lang_code"]) if norm["was_translated"] else txt
        result.append({"section": sec, "trust": SECTION_TRUST.get(sec, 0.5), "text_en": txt_en})
    return result


LANG_PAIR_PRIOR = {"fr": .95, "de": .95, "es": .95, "it": .94, "pt": .94, "nl": .93, "sv": .92, "pl": .88, "ru": .88,
                   "tr": .85, "id": .88, "vi": .83, "zh-cn": .82, "zh-tw": .80, "ja": .80, "ko": .80, "ar": .80,
                   "hi": .80, "ta": .75, "te": .75, "bn": .78, "ur": .78}


def translation_confidence(norm):
    if not norm["was_translated"]:
        return 1.0
    prior = LANG_PAIR_PRIOR.get(norm["detected_lang_code"], 0.8)
    o = [l.strip() for l in norm["original_text"].split("\n") if l.strip()]
    e = [l.strip() for l in norm["english_text"].split("\n") if l.strip()]
    unchanged = sum(1 for a, b in zip(o, e) if a == b) / max(len(o), 1)
    en = norm["english_text"]
    leftover = sum(ord(c) > 127 for c in en) / max(len(en), 1)
    return float(np.clip(prior * (1 - 0.5 * unchanged) * (1 - leftover), 0.3, 1.0))


def fusion_weights(conf):
    w_sem = 0.7 * conf
    return w_sem, 1.0 - w_sem


SKILL_TAXONOMY = {
    "Python": ("Python (computer programming)", ["python"]),
    "Java": ("Java (computer programming)", ["java"]),
    "JavaScript": ("JavaScript", ["javascript", "js"]),
    "Node.js / Web Application Development": ("web application development", ["node.js", "nodejs", "express", "express.js", "server pipelines"]),
    "React": ("React (web framework)", ["react", "react.js"]),
    "Flutter": ("Flutter / mobile app development", ["flutter", "dart"]),
    "SQL": ("SQL", ["sql", "pl/sql", "mysql", "postgresql"]),
    "Git": ("Git (version control)", ["git", "github"]),
    "C++": ("C++", ["c++"]),
    "HTML/CSS": ("HTML / CSS", ["html", "css"]),
    "AWS": ("cloud technologies", ["aws", "amazon web services", "ec2", "s3"]),
    "Docker": ("Docker / containerisation", ["docker", "containers", "kubernetes"]),
    "Machine Learning": ("machine learning", ["machine learning", "statistical modeling", "statistical modelling", "deep learning", "neural network", "scikit-learn", "tensorflow"]),
    "NLP": ("natural language processing", ["nlp", "natural language processing", "text mining"]),
    "Data Analysis": ("data analysis", ["data analysis", "pandas", "numpy", "data analytics", "statistics"]),
    "Firebase": ("Firebase", ["firebase"]),
    "GIS": ("geographic information systems", ["gis", "arcgis", "qgis"]),
    "Disaster Management": ("disaster management", ["disaster management", "ndma", "emergency response"]),
    "Cybersecurity": ("ICT security", ["cybersecurity", "cyber security", "encryption", "cryptography"]),
}
_ALIAS_RE = {c: re.compile("|".join(r"(?<![\w+])" + re.escape(a) + r"(?![\w+])" for a in al), re.I)
             for c, (_, al) in SKILL_TAXONOMY.items()}


def find_skills_direct(text):
    return {c for c, rx in _ALIAS_RE.items() if rx.search(text)}


def resolve_skills(sections, embedder=None, sim_threshold=0.55):
    found = {}

    def add(skill, trust, section, method):
        if skill not in found or trust > found[skill]["trust"]:
            found[skill] = {"trust": trust, "section": section, "method": method}

    frag_meta = []
    for s in sections:
        for sk in find_skills_direct(s["text_en"]):
            add(sk, s["trust"], s["section"], "alias")
        for line in s["text_en"].split("\n"):
            if len(line.strip()) > 25:
                frag_meta.append((line.strip(), s))
    if embedder is not None and frag_meta:
        labels = list(SKILL_TAXONOMY)
        lab_txt = [SKILL_TAXONOMY[c][0] + ": " + ", ".join(SKILL_TAXONOMY[c][1]) for c in labels]
        L = embedder.encode(lab_txt, normalize_embeddings=True)
        frags = frag_meta[:150]
        F = embedder.encode([f[0] for f in frags], normalize_embeddings=True)
        sims = F @ L.T
        for i, row in enumerate(sims):
            j = int(row.argmax())
            if row[j] >= sim_threshold:
                add(labels[j], frags[i][1]["trust"] * float(row[j]) ** 0.5, frags[i][1]["section"], "semantic")
    return found


@st.cache_data(show_spinner=False)
def esco_lookup(label, lang="en"):
    try:
        r = requests.get("https://ec.europa.eu/esco/api/search",
                         params={"text": label, "type": "skill", "language": lang, "limit": 1}, timeout=6)
        res = r.json().get("_embedded", {}).get("results", [])
        return (res[0].get("title"), res[0].get("uri")) if res else (None, None)
    except Exception:
        return (None, None)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(name):
    return SentenceTransformer(name)


def chunk_words(text, n=100, max_chunks=3):
    w = str(text).split()
    return [" ".join(w[i:i + n]) for i in range(0, max(len(w), 1), n)][:max_chunks] or [""]


@st.cache_resource(show_spinner="Embedding job corpus (one-time)...")
def embed_jobs(_embedder, model_name, job_docs):
    chunks, owner = [], []
    for i, d in enumerate(job_docs):
        for c in chunk_words(d):
            chunks.append(c)
            owner.append(i)
    E = _embedder.encode(chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    owner = np.array(owner)
    out = np.zeros((len(job_docs), E.shape[1]), dtype=np.float32)
    for i in range(len(job_docs)):
        v = E[owner == i].mean(0)
        out[i] = v / (np.linalg.norm(v) + 1e-9)
    return out


def trusted_resume_vector(embedder, sections):
    vecs, wts = [], []
    for s in sections:
        for c in chunk_words(s["text_en"], 100, 6):
            if c.strip():
                vecs.append(embedder.encode(c, normalize_embeddings=True))
                wts.append(s["trust"])
    if not vecs:
        return None
    v = np.average(np.array(vecs), axis=0, weights=wts)
    return v / (np.linalg.norm(v) + 1e-9)


def mine_pseudo_triplets(df, E, n=300):
    sims = E @ E.T
    trips = []
    for i in range(min(n, len(df))):
        order = np.argsort(-sims[i])
        neg = next((j for j in order[1:] if df.loc[j, "category"] != df.loc[i, "category"]), None)
        if neg is not None:
            trips.append((df.loc[i, "job_title"], df.loc[i, "job_doc"][:600], df.loc[neg, "job_doc"][:600]))
    return trips


def finetune_biencoder(model_name, triplets, epochs=1, batch=16):
    model = SentenceTransformer(model_name)
    ex = [InputExample(texts=[a, p, n]) for a, p, n in triplets]
    dl = DataLoader(ex, shuffle=True, batch_size=batch)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(dl, loss)], epochs=epochs, warmup_steps=10, show_progress_bar=False)
    model.save("careerai_finetuned")
    return "careerai_finetuned"


@st.cache_data(show_spinner=False)
def job_skill_sets(descs):
    return [find_skills_direct(d) for d in descs]


def fuse_scores(resume_vec, E_jobs, resume_skills, jobs_df, level, conf):
    w_sem, w_skill = fusion_weights(conf)
    sem = np.clip(E_jobs @ resume_vec, 0, 1)
    jsets = job_skill_sets(tuple(jobs_df["job_description"].astype(str).tolist()))
    rows = []
    for i, js in enumerate(jsets):
        n = max(len(js), 1)
        matched = {s: resume_skills[s]["trust"] / n for s in js if s in resume_skills}
        skill_part = w_skill * sum(matched.values())
        title = str(jobs_df.iloc[i]["job_title"]).lower()
        pen = 0.4 if level == "Entry-Level/Student" and any(k in title for k in ['senior', 'manager', 'director', 'lead', 'principal']) else 0.0
        rows.append((w_sem * sem[i], skill_part, pen, matched, js))
    out = jobs_df.copy()
    out["sem_part"] = [r[0] for r in rows]
    out["skill_part"] = [r[1] for r in rows]
    out["penalty"] = [r[2] for r in rows]
    out["adv_score"] = out["sem_part"] + out["skill_part"] - out["penalty"]
    out["matched"] = [r[3] for r in rows]
    out["job_skills"] = [r[4] for r in rows]
    mx = out["adv_score"].max()
    out["adv_pct"] = 50.0 if mx <= 0 else (out["adv_score"] / mx * 100).clip(0, 100)
    return out.sort_values("adv_score", ascending=False), (w_sem, w_skill, mx)


def explain_row(row):
    names, vals = ["Semantic title/content match"], [float(row["sem_part"])]
    total = max(sum(row["matched"].values()), 1e-9)
    for s, v in row["matched"].items():
        names.append(f"Skill: {s}")
        vals.append(float(v / total * row["skill_part"]))
    if row["penalty"] > 0:
        names.append("Seniority penalty")
        vals.append(-float(row["penalty"]))
    return names, vals


def render_waterfall(names, vals):
    if SHAP_OK:
        try:
            exp = shap.Explanation(values=np.array(vals), base_values=0.0,
                                   data=np.array([""] * len(vals), dtype=object), feature_names=names)
            shap.plots.waterfall(exp, show=False)
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
            return
        except Exception:
            pass
    fig = go.Figure(go.Waterfall(x=names, y=vals, measure=["relative"] * len(vals), text=[f"{v:+.3f}" for v in vals]))
    fig.update_layout(title="Score contributions")
    st.plotly_chart(fig, use_container_width=True)


def counterfactuals(row, w_skill, mx, resume_skills, jd_text, top_k=3):
    missing = [s for s in row["job_skills"] if s not in resume_skills]
    ranked = sorted(missing, key=lambda s: -len(_ALIAS_RE[s].findall(jd_text)))[:top_k]
    n = max(len(row["job_skills"]), 1)
    cur, steps = row["adv_score"], []
    for s in ranked:
        cur += w_skill * 1.0 / n
        steps.append((s, min(cur / mx * 100, 100)))
    return steps



# =========================================================
# MAIN APP
# =========================================================
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'tf_history' not in st.session_state:
    st.session_state.tf_history = None
if 'tf_pred' not in st.session_state:
    st.session_state.tf_pred = 0
if 'lang_info' not in st.session_state:
    st.session_state.lang_info = None


st.sidebar.header("⚙️ Controls")
cat_filter = st.sidebar.selectbox("Filter Industry", ["All"] + ALL_CATEGORIES)
if st.sidebar.button("🔄 Reset Analysis"):
    st.session_state.clear()
    st.rerun()
adv_model_label = st.sidebar.selectbox("🧬 Embedding model (Advanced tab)", list(EMBED_MODELS)) if ADV_OK else None


st.markdown("### 📤 Upload Your Resume")
st.caption("🌍 Supports resumes in any language — automatically detected and translated to English before analysis.")
uploaded = st.file_uploader("Choose PDF file", type="pdf")


if uploaded and st.button("🚀 Analyze Complete Resume", use_container_width=True, type="primary"):
    with st.spinner("🔍 Processing your resume..."):
        raw_text = extract_pdf_text(uploaded)


    if len(raw_text) < 50:
        st.error("❌ Resume too short or unreadable. Please use a text-based PDF.")
    else:
        with st.spinner("🌍 Detecting language and normalizing to English..."):
            norm = normalize_resume_to_english(raw_text)
        text = norm["english_text"]


        if not is_valid_resume(text):
            st.error("⚠️ Invalid document detected. Include contact information and at least two standard resume sections such as Experience, Education, or Skills.")
        else:
            skills, level = analyze_profile(text)
            matches = score_jobs(text, skills, level)
            st.session_state.analyzed = True
            st.session_state.matches = matches
            st.session_state.text = text
            st.session_state.original_text = norm["original_text"]
            st.session_state.skills = skills
            st.session_state.level = level
            st.session_state.lang_info = norm
            st.session_state.adv_sections = build_trusted_sections(uploaded, norm)
            st.session_state.trans_conf = translation_confidence(norm)
            st.session_state.pop("adv_result", None)
            if norm["was_translated"]:
                st.success(f"✅ Analysis complete. Detected {norm['detected_lang_name']}, translated to English • Found {len(skills)} skills • {level} profile")
            else:
                st.success(f"✅ Analysis complete. Found {len(skills)} skills • {level} profile")


if st.session_state.analyzed:
    if st.session_state.lang_info and st.session_state.lang_info["was_translated"]:
        st.markdown(
            f'<div class="lang-badge">🌍 Original language: {st.session_state.lang_info["detected_lang_name"]} → Translated to English for analysis</div>',
            unsafe_allow_html=True
        )
        with st.expander("👀 View original vs. translated text"):
            original_col, translated_col = st.columns(2)
            with original_col:
                st.markdown(f"**Original ({st.session_state.lang_info['detected_lang_name']})**")
                st.text_area("original_text", st.session_state.original_text[:2000], height=250, label_visibility="collapsed")
            with translated_col:
                st.markdown("**Translated (English)**")
                st.text_area("translated_text", st.session_state.text[:2000], height=250, label_visibility="collapsed")


    all_matches = st.session_state.matches.copy()
    if cat_filter != "All":
        filtered_matches = all_matches[all_matches['category'] == cat_filter]
        matches = filtered_matches.head(50) if not filtered_matches.empty else all_matches.head(50)
        if filtered_matches.empty:
            st.warning(f"No matches in {cat_filter}. Showing strongest overall alternatives.")
    else:
        matches = all_matches.head(50)


    if matches.empty:
        st.error("No matching jobs were found.")
        st.stop()


    best = matches.iloc[0]
    disp_sal = f"${best['est_salary']:,.0f}" if pd.notna(best['est_salary']) else "N/A"


    st.markdown(f"""
    <div class="kpi-card">
        <h2 style="margin:0 0 8px 0; color:#38bdf8;">{best['job_title'][:60]}{'...' if len(best['job_title']) > 60 else ''}</h2>
        <p style="margin:0 0 12px 0; color:#94a3b8; font-size:0.95em;">{best['company_name']} • {best['category']}</p>
        <div style="display:flex; gap:20px;">
            <span style="color:#4ade80; font-weight:bold; font-size:1.3em;">{best['match_pct']:.1f}% Match</span>
            <span style="color:#f59e0b;">Salary: {disp_sal}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("👤 Profile Level", st.session_state.level)
    metric_two.metric("🛠️ Detected Skills", len(st.session_state.skills))
    metric_three.metric("🔍 Jobs Analyzed", f"{len(matches):,}")


    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("#### 📋 Top Job Recommendations")
        show_hidden = st.checkbox("💎 Include Hidden Gems (non-obvious matches)")
        display_matches = matches.copy()
        if show_hidden and st.session_state.skills:
            pattern = '|'.join(re.escape(skill.lower()) for skill in st.session_state.skills)
            display_matches = display_matches[~display_matches['job_title'].str.lower().str.contains(pattern, na=False)]


        display_df = display_matches[['job_title', 'company_name', 'match_pct', 'est_salary']].head(10).copy()
        display_df['match_pct'] = display_df['match_pct'].map(lambda value: f"{value:.1f}%")
        display_df['est_salary'] = display_df['est_salary'].map(lambda value: f"${value:,.0f}" if pd.notna(value) else "N/A")
        st.dataframe(
            display_df.rename(columns={'job_title': '💼 Job Title', 'company_name': '🏢 Company', 'match_pct': '⭐ Match %', 'est_salary': '💰 Est. Salary'}),
            use_container_width=True,
            hide_index=True
        )


    with col_right:
        st.markdown("#### 📊 Category Distribution")
        counts = matches['category'].value_counts().head(6)
        if not counts.empty:
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
            _, _, autotexts = ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            plt.title("Top Matching Sectors", pad=20, fontsize=12, color='white')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Insufficient data for category chart.")


    st.markdown("---")
    t1, t2, t3, t4, t5, t6 = st.tabs(["🧠 Deep Learning Lab", "⚖️ Bias & Fairness", "📝 Cover Letters", "🗺️ Market Landscape", "🔥 Skills Heatmap", "🚀 Advanced Engine"])


    with t1:
        st.markdown("#### 🧠 Dual Model Salary Prediction")
        random_forest_col, neural_network_col = st.columns(2)
        with random_forest_col:
            st.markdown("**🤖 Random Forest (Scikit-Learn)**")
            if rf_model is not None:
                prediction = rf_model.predict(tfidf_vect.transform([st.session_state.text]))[0]
                st.metric("Predicted Salary", f"${prediction:,.0f}")
            else:
                st.info("Insufficient salary data to train the Random Forest model.")


        with neural_network_col:
            st.markdown("**⚡ Neural Network (TensorFlow)**")
            if st.button("🔄 Train NN Live", key="train_nn"):
                with st.spinner("Training Keras model..."):
                    tf_model, history, scalers = train_tf_model(df_jobs, X_jobs)
                    st.session_state.tf_history = history
                    if tf_model is not None:
                        user_dense = tfidf_vect.transform([st.session_state.text]).toarray()
                        pred_scaled = tf_model.predict(user_dense, verbose=0)[0][0]
                        st.session_state.tf_pred = (pred_scaled * scalers[1]) + scalers[0]
                    else:
                        st.warning("Insufficient salary data to train the Neural Network.")
            if st.session_state.tf_pred > 0:
                st.metric("NN Prediction", f"${st.session_state.tf_pred:,.0f}")


        if st.session_state.tf_history is not None:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=st.session_state.tf_history.history['loss'], name='Training Loss', line=dict(color='#3b82f6')))
            fig_loss.add_trace(go.Scatter(y=st.session_state.tf_history.history['val_loss'], name='Validation Loss', line=dict(color='#ef4444')))
            fig_loss.update_layout(title="Neural Network Convergence", showlegend=True)
            st.plotly_chart(fig_loss, use_container_width=True)


    with t2:
        st.markdown("#### ⚖️ Resume Language & Bias Analysis")
        strong_count, weak_count, label = audit_resume_bias(st.session_state.text)
        bias_one, bias_two, bias_three = st.columns(3)
        bias_one.metric("💪 Strong Action Verbs", strong_count)
        bias_two.metric("🙇 Passive Language", weak_count)
        bias_three.metric("📊 Tone Score", label, delta="Excellent" if strong_count > weak_count * 1.5 else "Good")
        if weak_count > strong_count * 1.2:
            st.warning('Replace phrases such as "helped with" with "led", "responsible for" with "managed", and "worked on" with "delivered".')
        else:
            st.success("✅ Excellent active language detected!")


    with t3:
        st.markdown("#### 📝 Professional Cover Letter Generator")
        st.caption("✨ Local templates • Category-aware")
        cover_col, output_col = st.columns([1, 3])
        with cover_col:
            if st.button("✨ Generate Cover Letter", use_container_width=True):
                st.session_state.generated_letter = cover_gen.generate(
                    job_title=best['job_title'],
                    company=best['company_name'],
                    job_desc=best['job_description'],
                    resume_text=st.session_state.text,
                    skills=st.session_state.skills,
                    level=st.session_state.level,
                    category=best['category']
                )
        with output_col:
            if 'generated_letter' in st.session_state:
                html_content = "<br>".join(line.strip() for line in st.session_state.generated_letter.splitlines())
                st.markdown(f'<div class="cover-letter">{html_content}</div>', unsafe_allow_html=True)
                st.download_button(" Download TXT", st.session_state.generated_letter, f"Cover_Letter_{best['company_name'].replace(' ', '_')}.txt", "text/plain")


    with t4:
        st.markdown("#### 🗺️ Your Position in Job Market")
        with st.spinner("Creating market visualization..."):
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_jobs.toarray())
            user_coords = pca.transform(tfidf_vect.transform([st.session_state.text]).toarray())
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_jobs)
            plot_df = df_jobs.copy()
            plot_df['PCA_X'] = coords[:, 0]
            plot_df['PCA_Y'] = coords[:, 1]
            plot_df['Cluster'] = clusters.astype(str)
            fig = px.scatter(plot_df, x='PCA_X', y='PCA_Y', color='category', hover_data=['job_title', 'company_name'], opacity=0.7, title="Your Resume vs. Job Market Landscape")
            fig.add_scatter(x=[user_coords[0, 0]], y=[user_coords[0, 1]], mode='markers+text', name='YOUR RESUME', marker=dict(size=18, color='red', symbol='star'), text=['YOU'], textposition="middle center", textfont=dict(size=14, color="white"))
            st.plotly_chart(fig, use_container_width=True)


    with t5:
        st.markdown("#### 🔥 Resume-Job Description Match Heatmap")
        job_words = set(best['job_description'].lower().split())
        lines = [line.strip() for line in st.session_state.text.split('\n') if len(line.strip()) > 15]
        for line in lines[:20]:
            match_count = len(set(line.lower().split()) & job_words)
            style = "heatmap-high" if match_count > 2 else "heatmap-low"
            marker = "⭐" if match_count > 2 else ""
            st.markdown(f'<div class="{style}">{marker} {line}</div>', unsafe_allow_html=True)


    with t6:
        st.markdown("#### 🚀 Dense Matching • Trust Fusion • XAI • ESCO Skills")
        if not ADV_OK:
            st.error("Advanced engine could not load. The exact error is below.")
            st.code(ADV_ERR)
            st.caption(f"Streamlit is running on this Python: {sys.executable}")
            st.markdown(
                "Install into **that** interpreter:\n\n"
                f"`\"{sys.executable}\" -m pip install sentence-transformers shap requests`\n\n"
                "If the error mentions DLL / protobuf / numpy, run:\n\n"
                f"`\"{sys.executable}\" -m pip install \"protobuf<5\" \"numpy<2\" --upgrade`"
            )
        else:
            conf = st.session_state.get("trans_conf", 1.0)
            w_sem_ui, w_skill_ui = fusion_weights(conf)
            c1, c2, c3 = st.columns(3)
            c1.metric("Translation confidence", f"{conf:.2f}")
            c2.metric("Semantic weight", f"{w_sem_ui:.2f}")
            c3.metric("Skill-entity weight", f"{w_skill_ui:.2f}")

            if st.button("▶ Run Advanced Engine", key="run_adv"):
                try:
                    with st.spinner("Embedding and fusing evidence (first run downloads the model)..."):
                        mname = EMBED_MODELS[adv_model_label]
                        emb = load_embedder(mname)
                        E_jobs = embed_jobs(emb, mname, tuple(df_jobs["job_doc"].tolist()))
                        sections = st.session_state.adv_sections
                        rv = trusted_resume_vector(emb, sections)
                        if rv is None:
                            st.error("No text available for embedding.")
                        else:
                            rskills = resolve_skills(sections, emb)
                            ranked, (ws, wk, mx) = fuse_scores(rv, E_jobs, rskills, df_jobs, st.session_state.level, conf)
                            st.session_state.adv_result = dict(ranked=ranked, rskills=rskills, wk=wk, mx=mx, E=E_jobs, name=mname)
                except Exception as e:
                    st.error(f"Advanced engine failed: {type(e).__name__}: {e}")

            res = st.session_state.get("adv_result")
            if res:
                ranked = res["ranked"]
                if cat_filter != "All" and (ranked["category"] == cat_filter).any():
                    ranked = ranked[ranked["category"] == cat_filter]
                top = ranked.head(10)
                st.dataframe(
                    top[["job_title", "company_name", "adv_pct", "sem_part", "skill_part", "penalty"]]
                    .rename(columns={"adv_pct": "Match %", "sem_part": "Semantic", "skill_part": "Skills", "penalty": "Seniority pen."}),
                    use_container_width=True, hide_index=True)

                pick = st.selectbox("Select a job to explain", list(top.index),
                                    format_func=lambda i: f"{top.loc[i, 'job_title'][:70]}")
                row = top.loc[pick]
                st.markdown(f"**Match: {row['adv_pct']:.1f}%**")
                names, vals = explain_row(row)
                render_waterfall(names, vals)

                steps = counterfactuals(row, res["wk"], res["mx"], res["rskills"], str(row["job_description"]))
                if steps:
                    msg = "; ".join(f"add **{s}** → {p:.0f}%" for s, p in steps)
                    st.info(f"Your resume matched this job at {row['adv_pct']:.0f}%. Adding the missing, requested skills (cumulative): {msg}.")
                else:
                    st.success("No missing taxonomy skills found for this job.")

                st.markdown("##### 🧩 Skills with evidence trust")
                skdf = pd.DataFrame([{"Skill": k, "Trust": round(v["trust"], 2), "Found in": v["section"], "Method": v["method"]}
                                     for k, v in res["rskills"].items()])
                if not skdf.empty:
                    skdf = skdf.sort_values("Trust", ascending=False)
                    if st.checkbox("Resolve official ESCO URIs (needs internet)"):
                        look = [esco_lookup(SKILL_TAXONOMY[s][0]) for s in skdf["Skill"]]
                        skdf["ESCO label"] = [l[0] for l in look]
                        skdf["ESCO URI"] = [l[1] for l in look]
                    st.dataframe(skdf, use_container_width=True, hide_index=True)
                else:
                    st.info("No taxonomy skills detected.")

            with st.expander("🎯 Contrastive fine-tuning with runner-up hard negatives"):
                st.caption("CSV columns: anchor, positive, hard_negative (hard_negative = the near-miss / runner-up example).")
                up = st.file_uploader("Transitions CSV", type="csv", key="hn_csv")
                if st.button("Fine-tune bi-encoder"):
                    try:
                        with st.spinner("Training..."):
                            mname = EMBED_MODELS[adv_model_label]
                            if up is not None:
                                t = pd.read_csv(up)
                                trips = list(t[["anchor", "positive", "hard_negative"]].itertuples(index=False, name=None))
                            else:
                                E = res["E"] if res else embed_jobs(load_embedder(mname), mname, tuple(df_jobs["job_doc"].tolist()))
                                trips = mine_pseudo_triplets(df_jobs, E)
                                st.warning("No CSV uploaded: used pseudo hard negatives (demo only).")
                            path = finetune_biencoder(mname, trips)
                        st.success(f"Saved to ./{path}. Add it to EMBED_MODELS to use it.")
                    except Exception as e:
                        st.error(f"Fine-tuning failed: {type(e).__name__}: {e}")


st.markdown("---")
st.markdown("### 📊 Powered by: TensorFlow • Scikit-Learn • langdetect + deep-translator • Sentence-Transformers • SHAP")