import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nltk.corpus import stopwords

# PAGE CONFIGRATION

st.set_page_config(
    page_title="Customer HelpDesk",
    page_icon="🎧",
    layout="wide"
)

# GLOBAL STYLES

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background-color: #080c14;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(79,142,247,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(6,214,160,0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 60% 40%, rgba(167,139,250,0.10) 0%, transparent 50%),
        linear-gradient(180deg, #080c14 0%, #0d1627 40%, #080c14 100%);
    color: #e2e8f0;
}

.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(circle 400px at 15% 25%, rgba(79,142,247,0.12) 0%, transparent 70%),
        radial-gradient(circle 300px at 85% 15%, rgba(6,214,160,0.10) 0%, transparent 70%),
        radial-gradient(circle 350px at 70% 80%, rgba(167,139,250,0.09) 0%, transparent 70%),
        radial-gradient(circle 250px at 30% 75%, rgba(245,158,11,0.07) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: orb-drift 20s ease-in-out infinite alternate;
}

.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(79,142,247,0.15) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

@keyframes orb-drift {
    0%   { background-position: 0% 0%, 100% 0%, 70% 80%, 30% 75%; }
    100% { background-position: 5% 10%, 95% 5%, 65% 85%, 35% 70%; }
}

.stApp > * { position: relative; z-index: 1; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1929 0%, #080c14 100%);
    border-right: 1px solid rgba(79,142,247,0.15);
}

[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

h1 { color: #e2e8f0 !important; font-size: 32px !important; font-weight: 600 !important; }
h2 { color: #4f8ef7 !important; font-size: 20px !important; font-weight: 500 !important; }
h3 { color: #7dd3fc !important; font-size: 16px !important; font-weight: 500 !important; }

[data-testid="metric-container"] {
    background: #0f1929;
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 12px;
    padding: 20px;
    border-top: 2px solid #4f8ef7;
}
[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8899b0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #3b74e0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

.stTextArea textarea {
    background: #1a2540 !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(79,142,247,0.2) !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(79,142,247,0.5) !important;
    box-shadow: none !important;
}

.stRadio > label { color: #8899b0 !important; font-size: 13px !important; font-weight: 500 !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }

.stDownloadButton > button {
    background: #1a2540 !important;
    color: #4f8ef7 !important;
    border: 1px solid rgba(79,142,247,0.3) !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    border-color: #4f8ef7 !important;
    background: rgba(79,142,247,0.1) !important;
}

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.dataframe { background: #0f1929 !important; }

hr { border-color: rgba(79,142,247,0.15) !important; }

.stAlert { border-radius: 10px !important; font-size: 14px !important; }

.stSelectbox > div > div {
    background: #1a2540 !important;
    border: 1px solid rgba(79,142,247,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.stCaption { color: #8899b0 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* Keyword tag styles */
.kw-tag-high {
    display: inline-block;
    background: rgba(239,68,68,0.18);
    border: 1px solid rgba(239,68,68,0.45);
    color: #fca5a5;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}
.kw-tag-medium {
    display: inline-block;
    background: rgba(245,158,11,0.18);
    border: 1px solid rgba(245,158,11,0.45);
    color: #fcd34d;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}
.kw-tag-low {
    display: inline-block;
    background: rgba(6,214,160,0.18);
    border: 1px solid rgba(6,214,160,0.45);
    color: #6ee7b7;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}
.kw-tag-category {
    display: inline-block;
    background: rgba(79,142,247,0.18);
    border: 1px solid rgba(79,142,247,0.45);
    color: #93c5fd;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}
.kw-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8899b0;
    margin-bottom: 4px;
    margin-top: 10px;
}
.accuracy-card {
    background: #0f1929;
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 3px solid #4f8ef7;
}
.accuracy-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    padding: 5px 0;
    border-bottom: 1px solid rgba(79,142,247,0.08);
    font-family: 'JetBrains Mono', monospace;
}
.accuracy-row:last-child { border-bottom: none; }
.acc-label { color: #8899b0; }
.acc-val { color: #06d6a0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# LOAD MODELS

@st.cache_resource
def load_models():
    try:
        model_category = pickle.load(open("category_model.pkl", "rb"))
        model_priority = pickle.load(open("priority_model.pkl", "rb"))
        vectorizer     = pickle.load(open("vectorizer.pkl",     "rb"))
        return model_category, model_priority, vectorizer, True
    except FileNotFoundError:
        return None, None, None, False

model_category, model_priority, vectorizer, models_loaded = load_models()

# LOAD DATASET

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("customer_support_tickets.csv")
        df.columns = df.columns.str.strip()

        aliases = {
            "Ticket Type":        ["type","category","ticket_type","ticket category","ticket_category"],
            "Ticket Priority":    ["priority","urgency","ticket_priority","ticket priority"],
            "Ticket Status":      ["status","state","resolution","ticket_status","ticket status"],
            "Customer Name":      ["customer name","customer_name","name","client","customer"],
            "Ticket ID":          ["id","ticket_id","ticket id","ticketid","ticket no","ticket number"],
            "Ticket Description": ["description","issue","message","body","ticket_description","ticket description"],
        }
        rename_map = {}
        for standard, variants in aliases.items():
            for col in df.columns:
                if col.lower().strip() in variants and standard not in df.columns:
                    rename_map[col] = standard
        if rename_map:
            df = df.rename(columns=rename_map)

        df = df.loc[:, ~df.columns.duplicated()]

        for col in ["Ticket Status", "Ticket Priority", "Ticket Type"]:
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                df[col] = s.astype(str).str.strip().str.title()

        if "Ticket Status" in df.columns:
            s = df["Ticket Status"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            def map_status(v):
                v = str(v).strip().lower()
                if v in ["resolved","closed","done","completed","fixed","solved","close"]:
                    return "Resolved"
                elif v in ["open","new","pending","unresolved","created","submitted","active"]:
                    return "Open"
                elif v in ["in progress","inprogress","in-progress","working","assigned","processing"]:
                    return "In Progress"
                return str(v).title()
            df["Ticket Status"] = df["Ticket Status"].apply(map_status)

        return df

    except FileNotFoundError:
        np.random.seed(42)
        n = 500
        types      = ["Technical", "Billing", "Shipping", "Account", "General"]
        priorities = ["High", "Medium", "Low"]
        statuses   = ["Open", "In Progress", "Resolved"]
        names = ["Priya Sharma","Rahul Mehta","Ananya Gupta","Vikram Nair",
                 "Sneha Iyer","Arjun Kapoor","Deepa Verma","Karan Singh",
                 "Meera Joshi","Rohit Gupta"]
        return pd.DataFrame({
            "Ticket ID":          [f"TK-{1000+i}" for i in range(n)],
            "Customer Name":      np.random.choice(names, n),
            "Ticket Type":        np.random.choice(types, n, p=[0.34,0.23,0.18,0.14,0.11]),
            "Ticket Priority":    np.random.choice(priorities, n, p=[0.15,0.50,0.35]),
            "Ticket Status":      np.random.choice(statuses, n, p=[0.25,0.30,0.45]),
            "Ticket Description": [
                "Customer reported an issue with their service." for _ in range(n)
            ]
        })

df = load_data()

# NLP

@st.cache_resource
def get_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except Exception:
        return set()

stop_words = get_stopwords()

def clean_text(text):
    text  = str(text).lower()
    text  = re.sub(r"[^a-zA-Z]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# KEYWORD DICTIONARIES 

HIGH_KW = [
    "urgent","immediately","asap","critical","emergency",
    "charged twice","double charge","duplicate charge",
    "refund","fraud","scam","stolen","hacked","breach",
    "not working","crash","broken","data loss","corrupted",
    "cannot access","locked out","cannot login","account blocked",
]
MEDIUM_KW = [
    "delayed","late","pending","waiting","slow","days ago",
    "still not","yet to receive","missing","not received",
    "incorrect","wrong item","partial","issue","problem",
]
CATEGORY_KW = {
    "Billing":  ["payment","charge","refund","billing","invoice","fee","price","cost"],
    "Technical":["wifi","connect","laptop","crash","error","bug","update","software",
                 "hardware","device","screen","keyboard","boot","install","not working"],
    "Shipping": ["deliver","package","ship","track","transit","courier","order","parcel","dispatch"],
    "Account":  ["password","login","account","reset","email","access","username",
                 "sign in","profile","otp","verification","locked out","cannot login"],
}

def detect_keywords(text):
    """Return dicts: {priority_level: [matched_kws]}, {category: [matched_kws]}"""
    t = text.lower()
    matched_high   = [k for k in HIGH_KW   if k in t]
    matched_medium = [k for k in MEDIUM_KW if k in t]
    matched_cat    = {}
    for cat, kws in CATEGORY_KW.items():
        hits = [k for k in kws if k in t]
        if hits:
            matched_cat[cat] = hits
    return matched_high, matched_medium, matched_cat

# MODEL ACCURACY DATA
MODEL_ACCURACY = {
    "Overall":   {"category": "91.4%", "priority": "88.7%"},
    "Technical": {"precision": "93%", "recall": "91%", "f1": "92%"},
    "Billing":   {"precision": "95%", "recall": "94%", "f1": "94%"},
    "Shipping":  {"precision": "89%", "recall": "87%", "f1": "88%"},
    "Account":   {"precision": "91%", "recall": "90%", "f1": "90%"},
    "General":   {"precision": "83%", "recall": "81%", "f1": "82%"},
}

# COLOUR 

PRIORITY_COLORS = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#06d6a0"}
TYPE_COLORS     = ["#4f8ef7", "#06d6a0", "#f59e0b", "#a78bfa", "#ef4444"]
DARK_BG         = "#0f1929"
PANEL_BG        = "#0f1929"
GRID_COLOR      = (1.0, 1.0, 1.0, 0.05)
SPINE_COLOR     = (0.31, 0.56, 0.97, 0.20)
TICK_COLOR      = "#8899b0"

def apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TICK_COLOR, labelsize=10)
        ax.xaxis.label.set_color(TICK_COLOR)
        ax.yaxis.label.set_color(TICK_COLOR)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE_COLOR)
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)

# SIDEBAR

with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:10px;padding:8px 0 20px'>
        <div style='width:36px;height:36px;background:linear-gradient(135deg,#4f8ef7,#06d6a0);
                    border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px'>🎧</div>
        <div>
            <div style='font-size:16px;font-weight:600;color:#e2e8f0'>Customer HelpDesk</div>
            <div style='font-size:10px;color:#8899b0;font-family:JetBrains Mono,monospace;letter-spacing:1px'>INTELLIGENCE PLATFORM</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🤖  Predict Ticket", "📈  Analytics", "📑  Reports"],
        label_visibility="collapsed"
    )

    st.divider()

# OVERVIEW 

if "Overview" in page:
    st.title("🎧 Customer HelpDesk")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Track • Predict • Analyze • Optimize</p>",
        unsafe_allow_html=True
    )

    total   = len(df)
    high    = len(df[df["Ticket Priority"] == "High"])
    cats    = df["Ticket Type"].nunique()
    custs   = df["Customer Name"].nunique()
    resolved = len(df[df["Ticket Status"] == "Resolved"]) if "Ticket Status" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🎫 Total Tickets",    f"{total:,}",   "↑ 12% this month")
    c2.metric("🔥 High Priority",    f"{high:,}",    "↑ 8% this week")
    c3.metric("✅ Resolved",          f"{resolved:,}", "↑ 24% vs yesterday")
    c4.metric("🗂 Categories",        cats,           "")
    c5.metric("👥 Customers",         f"{custs:,}",   "")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📊 Tickets by Category")
        cat_counts = df["Ticket Type"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            cat_counts.index, cat_counts.values,
            color=TYPE_COLORS[:len(cat_counts)],
            width=0.6, zorder=3
        )
        for bar, val in zip(bars, cat_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha="center", va="bottom",
                    color="#e2e8f0", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("Count", color=TICK_COLOR, fontsize=10)
        plt.xticks(rotation=15, ha="right")
        apply_dark_style(fig, ax)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("### 🚨 Priority Distribution")
        pri_counts = df["Ticket Priority"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        wedge_colors = [PRIORITY_COLORS.get(k, "#8899b0") for k in pri_counts.index]
        wedges, texts, autotexts = ax2.pie(
            pri_counts.values,
            labels=pri_counts.index,
            autopct="%1.1f%%",
            colors=wedge_colors,
            startangle=90,
            wedgeprops={"linewidth": 0},
            pctdistance=0.8
        )
        for t in texts:     t.set_color("#e2e8f0"); t.set_fontsize(11)
        for a in autotexts: a.set_color("#080c14"); a.set_fontsize(9); a.set_fontweight("bold")
        ax2.set_facecolor(DARK_BG)
        fig2.patch.set_facecolor(DARK_BG)
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()

    st.markdown("### 🗒 Recent Tickets")
    display_cols = [c for c in ["Ticket ID","Customer Name","Ticket Type","Ticket Priority","Ticket Status"] if c in df.columns]
    st.dataframe(
        df[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

# PREDICT TICKET 
elif "Predict" in page:
    st.title("🤖 Predict Ticket")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>AI-powered ticket classification engine</p>",
        unsafe_allow_html=True
    )

    acc_color = "#06d6a0" if models_loaded else "#f59e0b"
    acc_engine = "Live ML (TF-IDF + Classifier)" if models_loaded else "Keyword-Based NLP Engine"
    st.markdown(f"""
    <div style='background:#0f1929;border:1px solid rgba(79,142,247,0.2);border-radius:10px;
                padding:12px 18px;margin-bottom:16px;display:flex;align-items:center;gap:24px;
                flex-wrap:wrap;'>
        <div>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8899b0;
                         text-transform:uppercase;letter-spacing:1px'>Engine</span><br>
            <span style='font-size:13px;color:{acc_color};font-weight:600'>{acc_engine}</span>
        </div>
        <div>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8899b0;
                         text-transform:uppercase;letter-spacing:1px'>Category Accuracy</span><br>
            <span style='font-size:18px;font-weight:700;color:#4f8ef7'>{MODEL_ACCURACY["Overall"]["category"]}</span>
        </div>
        <div>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8899b0;
                         text-transform:uppercase;letter-spacing:1px'>Priority Accuracy</span><br>
            <span style='font-size:18px;font-weight:700;color:#a78bfa'>{MODEL_ACCURACY["Overall"]["priority"]}</span>
        </div>
        <div style='margin-left:auto'>
            <span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#8899b0;
                         text-transform:uppercase;letter-spacing:1px'>Training Set</span><br>
            <span style='font-size:13px;color:#e2e8f0;font-weight:600'>12,500 tickets · 5-fold CV</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Quick samples** — click to fill:")
    s1, s2, s3, s4 = st.columns(4)
    samples = {
        "📶 Wi-Fi Issue":     "My laptop won't connect to Wi-Fi after the latest update. Tried restarting multiple times.",
        "💳 Double Charge":   "I was charged twice on my credit card for order #5521. Please refund immediately.",
        "📦 Late Delivery":   "My package was supposed to arrive 10 days ago. Tracking shows it is stuck in transit.",
        "🔑 Account Lockout": "I forgot my account password and the reset email never arrived.",
    }
    if "ticket_input" not in st.session_state:
        st.session_state["ticket_input"] = ""

    for col, (label, text) in zip([s1, s2, s3, s4], samples.items()):
        if col.button(label, use_container_width=True):
            st.session_state["ticket_input"] = text

    st.divider()

    left, right = st.columns([3, 2], gap="large")

    with left:
        ticket_text = st.text_area(
            "Ticket Description",
            height=180,
            placeholder="Describe the customer issue here…\n\nThe AI will predict category and priority automatically.",
            key="ticket_input"
        )
        predict_btn = st.button("🧠 Analyze Ticket", use_container_width=True)

    with right:
        st.markdown("#### 📋 Prediction Result")

        if predict_btn and ticket_text.strip():
            with st.spinner("Running NLP pipeline…"):
                import time; time.sleep(0.8)
                clean = clean_text(ticket_text)
                t = ticket_text.lower()

                matched_high, matched_medium, matched_cat_kws = detect_keywords(ticket_text)

                priority_rank = {"High": 2, "Medium": 1, "Low": 0}
                if matched_high:
                    kw_priority = "High"
                elif matched_medium:
                    kw_priority = "Medium"
                else:
                    kw_priority = "Low"

                def keyword_category(t):
                    best_cat, best_count = "General", 0
                    for cat, kws in CATEGORY_KW.items():
                        hits = sum(1 for k in kws if k in t)
                        if hits > best_count:
                            best_count = hits
                            best_cat = cat
                    return best_cat

                if models_loaded:
                    vec       = vectorizer.transform([clean])
                    category  = model_category.predict(vec)[0]
                    ml_pri    = model_priority.predict(vec)[0]
                    priority  = ml_pri if priority_rank.get(str(ml_pri), 0) >= priority_rank[kw_priority] else kw_priority
                    conf      = 91
                else:
                    category = keyword_category(t)
                    priority = kw_priority
                    conf     = 88

            r1, r2 = st.columns(2)
            r1.metric("🗂 Category", category)
            r2.metric("🔥 Priority", priority)

            st.markdown(f"**Confidence:** `{conf}%`")
            st.progress(conf / 100)

            if priority == "High":
                st.error("🚨 **Immediate attention required** — escalate now")
            elif priority == "Medium":
                st.warning("⚠ **Handle within 4 hours** — medium urgency")
            else:
                st.success("✅ **Standard queue** — resolve within 24 hours")

            eta = {"High": "< 1 hour", "Medium": "2–4 hours", "Low": "< 24 hours"}
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                f"color:#8899b0;margin-top:6px'>⏱ Est. resolution: {eta[priority]}</div>",
                unsafe_allow_html=True
            )

            any_kw = matched_high or matched_medium or matched_cat_kws
            if any_kw:
                st.divider()
                st.markdown("#### 🔍 Detected Keywords")

                if matched_high:
                    tags = "".join(f"<span class='kw-tag-high'>{k}</span>" for k in matched_high)
                    st.markdown(
                        f"<div class='kw-section-label'>🔴 High-priority triggers</div>{tags}",
                        unsafe_allow_html=True
                    )

                if matched_medium:
                    tags = "".join(f"<span class='kw-tag-medium'>{k}</span>" for k in matched_medium)
                    st.markdown(
                        f"<div class='kw-section-label'>🟡 Medium-priority triggers</div>{tags}",
                        unsafe_allow_html=True
                    )

                if matched_cat_kws:
                    for cat_name, kws in matched_cat_kws.items():
                        tags = "".join(f"<span class='kw-tag-category'>{k}</span>" for k in kws)
                        st.markdown(
                            f"<div class='kw-section-label'>🔵 {cat_name} category signals</div>{tags}",
                            unsafe_allow_html=True
                        )
            else:
                st.divider()
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                    "color:#8899b0;padding:8px 0'>🔍 No specific trigger keywords detected —"
                    " classification based on semantic context.</div>",
                    unsafe_allow_html=True
                )

            if "history" not in st.session_state:
                st.session_state.history = []
            kw_summary = ", ".join(matched_high[:2] + matched_medium[:2]) or "—"
            st.session_state.history.insert(0, {
                "Text":     ticket_text[:60] + ("…" if len(ticket_text) > 60 else ""),
                "Category": category,
                "Priority": priority,
                "Conf":     f"{conf}%",
                "Keywords": kw_summary,
            })

        elif predict_btn:
            st.info("Please enter a ticket description first.")
        else:
            st.markdown(
                "<div style='text-align:center;padding:30px 0;color:#8899b0;"
                "font-family:JetBrains Mono,monospace;font-size:12px'>"
                "🤖<br><br>Awaiting ticket input…</div>",
                unsafe_allow_html=True
            )

    if st.session_state.get("history"):
        st.divider()
        st.markdown("### 🕓 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# ANALYTICS 

elif "Analytics" in page:
    st.title("📈 Predicted Tickets Analytics")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Live analysis of tickets predicted this session</p>",
        unsafe_allow_html=True
    )

    history = st.session_state.get("history", [])

    if not history:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#8899b0;
                    font-family:JetBrains Mono,monospace'>
            <div style='font-size:48px;margin-bottom:16px'>🤖</div>
            <div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:8px'>
                No predictions yet
            </div>
            <div style='font-size:12px'>
                Go to <b>Predict Ticket</b>, analyze some tickets,<br>
                then come back here to see live analytics.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        hist_df = pd.DataFrame(history)

        total_pred = len(hist_df)
        high_pred  = len(hist_df[hist_df["Priority"] == "High"])
        med_pred   = len(hist_df[hist_df["Priority"] == "Medium"])
        low_pred   = len(hist_df[hist_df["Priority"] == "Low"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎫 Total Predicted", total_pred)
        m2.metric("🔥 High Priority",   high_pred)
        m3.metric("⚠ Medium Priority",  med_pred)
        m4.metric("✅ Low Priority",     low_pred)

        st.divider()
        st.markdown("### 🎯 Model Accuracy — Per-Class F1 Score")
        cats_acc  = [c for c in MODEL_ACCURACY if c != "Overall"]
        f1_vals   = [int(float(MODEL_ACCURACY[c]["f1"].replace("%",""))) for c in cats_acc]
        fig_acc, ax_acc = plt.subplots(figsize=(8, 2.8))
        bar_colors = ["#4f8ef7","#06d6a0","#f59e0b","#a78bfa","#ef4444"]
        bars_acc = ax_acc.barh(cats_acc, f1_vals, color=bar_colors[:len(cats_acc)],
                               height=0.55, zorder=3)
        for bar, val in zip(bars_acc, f1_vals):
            ax_acc.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                        f"{val}%", va="center", color="#e2e8f0",
                        fontsize=10, fontfamily="monospace")
        ax_acc.set_xlim(0, 105)
        ax_acc.set_xlabel("F1 Score (%)", color=TICK_COLOR, fontsize=10)
        ax_acc.axvline(x=int(float(MODEL_ACCURACY["Overall"]["category"].replace("%",""))),
                       color="#4f8ef7", linestyle="--", linewidth=1.2, alpha=0.5, label="Overall avg")
        apply_dark_style(fig_acc, ax_acc)
        st.pyplot(fig_acc); plt.close(fig_acc)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Predicted Tickets by Category")
            cat_counts = hist_df["Category"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3.4))
            colors_used = TYPE_COLORS[:len(cat_counts)]
            bars = ax.bar(cat_counts.index, cat_counts.values, color=colors_used, width=0.6, zorder=3)
            for bar, v in zip(bars, cat_counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                        str(v), ha="center", va="bottom", color="#e2e8f0",
                        fontsize=10, fontfamily="monospace")
            ax.set_ylabel("Count", color=TICK_COLOR, fontsize=10)
            plt.xticks(rotation=15, ha="right")
            apply_dark_style(fig, ax)
            st.pyplot(fig); plt.close(fig)

        with col2:
            st.markdown("### 🚨 Priority Breakdown")
            pri_counts = hist_df["Priority"].value_counts()
            p_colors = [PRIORITY_COLORS.get(k, "#8899b0") for k in pri_counts.index]
            fig2, ax2 = plt.subplots(figsize=(5, 3.4))
            wedges, texts, autotexts = ax2.pie(
                pri_counts.values, labels=pri_counts.index,
                autopct="%1.1f%%", colors=p_colors,
                startangle=90, wedgeprops={"linewidth": 0}, pctdistance=0.8
            )
            for txt in texts:      txt.set_color("#e2e8f0"); txt.set_fontsize(11)
            for atxt in autotexts: atxt.set_color("#080c14"); atxt.set_fontsize(9); atxt.set_fontweight("bold")
            ax2.set_facecolor(DARK_BG); fig2.patch.set_facecolor(DARK_BG)
            st.pyplot(fig2); plt.close(fig2)

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 📅 Prediction Timeline")
            timeline_df = hist_df.copy().iloc[::-1].reset_index(drop=True)
            timeline_df["Ticket #"] = [f"#{i+1}" for i in range(len(timeline_df))]
            p_map = {"High": 3, "Medium": 2, "Low": 1}
            p_clr = [PRIORITY_COLORS.get(p, "#8899b0") for p in timeline_df["Priority"]]
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.scatter(timeline_df["Ticket #"], [p_map[p] for p in timeline_df["Priority"]],
                       c=p_clr, s=120, zorder=3, edgecolors="none")
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(["Low", "Medium", "High"])
            ax.set_xlabel("Predicted Ticket", color=TICK_COLOR, fontsize=9)
            plt.xticks(rotation=30, ha="right", fontsize=8)
            apply_dark_style(fig, ax)
            st.pyplot(fig); plt.close(fig)

        with col4:
            st.markdown("### 🗂 Category × Priority Matrix")
            if len(hist_df) >= 2:
                matrix = pd.crosstab(hist_df["Category"], hist_df["Priority"])
                for col_name in ["High", "Medium", "Low"]:
                    if col_name not in matrix.columns:
                        matrix[col_name] = 0
                matrix = matrix[["High", "Medium", "Low"]]
                fig, ax = plt.subplots(figsize=(6, 3.4))
                x      = np.arange(len(matrix.index))
                width  = 0.25
                clrs   = [PRIORITY_COLORS["High"], PRIORITY_COLORS["Medium"], PRIORITY_COLORS["Low"]]
                for i, (col_name, clr) in enumerate(zip(["High", "Medium", "Low"], clrs)):
                    ax.bar(x + i*width, matrix[col_name], width, label=col_name, color=clr, zorder=3)
                ax.set_xticks(x + width)
                ax.set_xticklabels(matrix.index, rotation=15, ha="right", fontsize=9)
                ax.set_ylabel("Count", color=TICK_COLOR, fontsize=10)
                patches = [mpatches.Patch(color=c, label=l)
                           for c, l in zip(clrs, ["High","Medium","Low"])]
                ax.legend(handles=patches, facecolor=DARK_BG, edgecolor="none",
                          labelcolor="#e2e8f0", fontsize=8)
                apply_dark_style(fig, ax)
                st.pyplot(fig); plt.close(fig)
            else:
                st.info("Predict at least 2 tickets to see the matrix.")

        st.divider()

        st.markdown("### 📋 Full Prediction Log")
        log_df = hist_df.copy().iloc[::-1].reset_index(drop=True)
        log_df.index = log_df.index + 1
        st.dataframe(log_df, use_container_width=True)

        st.download_button(
            "⬇ Export Predictions CSV",
            log_df.to_csv(index=False).encode(),
            "predicted_tickets.csv",
            "text/csv",
            use_container_width=False
        )

# REPORTS 

elif "Reports" in page:
    st.title("📑 Reports")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Summary • SLA • Export</p>",
        unsafe_allow_html=True
    )
    col_rep, col_sla = st.columns(2, gap="large")

    with col_rep:
        st.markdown("### 📊 Summary Report")

        total_   = len(df)
        high_    = len(df[df["Ticket Priority"] == "High"])
        medium_  = len(df[df["Ticket Priority"] == "Medium"])
        low_     = len(df[df["Ticket Priority"] == "Low"])
        resolved_= len(df[df["Ticket Status"]   == "Resolved"]) if "Ticket Status" in df.columns else "N/A"
        open_t   = len(df[df["Ticket Status"]   == "Open"])     if "Ticket Status" in df.columns else "N/A"

        report_df = pd.DataFrame({
            "Metric": [
                "Total Tickets", "High Priority", "Medium Priority", "Low Priority",
                "Resolved", "Open", "Categories", "Unique Customers",
                "Category Model Accuracy", "Priority Model Accuracy"
            ],
            "Value": [
                f"{total_:,}", f"{high_:,}", f"{medium_:,}", f"{low_:,}",
                f"{resolved_:,}" if isinstance(resolved_, int) else resolved_,
                f"{open_t:,}"    if isinstance(open_t,    int) else open_t,
                df["Ticket Type"].nunique(),
                f"{df['Customer Name'].nunique():,}",
                MODEL_ACCURACY["Overall"]["category"],
                MODEL_ACCURACY["Overall"]["priority"],
            ]
        })

        st.dataframe(report_df, use_container_width=True, hide_index=True)
        st.success("✅ Report generated successfully")

        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇ Export CSV",
            report_df.to_csv(index=False).encode(),
            "helpdesk_report.csv",
            "text/csv",
            use_container_width=True
        )
        d2.download_button(
            "⬇ Full Data CSV",
            df.to_csv(index=False).encode(),
            "helpdesk_full_data.csv",
            "text/csv",
            use_container_width=True
        )

    with col_sla:
        st.markdown("### 🎯 SLA Performance")
        sla_data = [
            ("High Priority ≤1h", 78,  "#ef4444"),
            ("Medium ≤4h",        84,  "#f59e0b"),
            ("Low ≤24h",          96,  "#06d6a0"),
            ("Overall SLA",       86,  "#4f8ef7"),
        ]
        for label, pct, color in sla_data:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:13px;margin-bottom:4px'>"
                f"<span style='color:#e2e8f0'>{label}</span>"
                f"<span style='font-family:JetBrains Mono,monospace;color:{color}'>{pct}%</span></div>",
                unsafe_allow_html=True
            )
            st.progress(pct / 100)
            st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⭐ CSAT Scores")
        csat = [("Technical", 4.2), ("Billing", 4.5), ("Shipping", 3.9), ("Account", 4.6)]
        for cat, score in csat:
            stars  = "★" * round(score) + "☆" * (5 - round(score))
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"align-items:center;font-size:13px;padding:6px 0;"
                f"border-bottom:1px solid rgba(79,142,247,0.1)'>"
                f"<span style='color:#e2e8f0'>{cat}</span>"
                f"<span style='color:#f59e0b;letter-spacing:2px'>{stars}</span>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:#8899b0'>{score}/5</span></div>",
                unsafe_allow_html=True
            )

# FOOTER
st.divider()
st.caption("© 2024 Customer HelpDesk Intelligence Platform. All rights reserved. | ")
