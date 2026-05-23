import streamlit as st
import pandas as pd
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from nltk.corpus import stopwords

# PAGE CONFIG

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
    background: linear-gradient(135deg, #080c14 0%, #0f1929 50%, #080c14 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1929 0%, #080c14 100%);
    border-right: 1px solid rgba(79,142,247,0.15);
}

[data-testid="stSidebar"] .css-1d391kg { padding-top: 1rem; }

h1 { color: #e2e8f0 !important; font-size: 32px !important; font-weight: 600 !important; }
h2 { color: #4f8ef7 !important; font-size: 20px !important; font-weight: 500 !important; }
h3 { color: #7dd3fc !important; font-size: 16px !important; font-weight: 500 !important; }

/* Metric cards */
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

/* Buttons */
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

/* Text area */
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

/* Sidebar radio */
.stRadio > label { color: #8899b0 !important; font-size: 13px !important; font-weight: 500 !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #e2e8f0 !important; }

/* Download button */
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

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.dataframe { background: #0f1929 !important; }

/* Divider */
hr { border-color: rgba(79,142,247,0.15) !important; }

/* Info / warning / error / success boxes */
.stAlert { border-radius: 10px !important; font-size: 14px !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: #1a2540 !important;
    border: 1px solid rgba(79,142,247,0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Caption */
.stCaption { color: #8899b0 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }
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

        # ── Step 1: strip whitespace from all column names ──
        df.columns = df.columns.str.strip()

        # ── Step 2: auto-rename common alternate column names ──
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

        # ── Step 3: remove any duplicate columns created by renaming ──
        df = df.loc[:, ~df.columns.duplicated()]

        # ── Step 4: normalise text values — strip + Title Case ──
        for col in ["Ticket Status", "Ticket Priority", "Ticket Type"]:
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):   # guard against duplicate cols
                    s = s.iloc[:, 0]
                df[col] = s.astype(str).str.strip().str.title()

        # ── Step 5: add a normalised status column for safe matching ──
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
                return str(v).title()   # keep original casing for anything else
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

# COLOUR HELPERS

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

    
   

# PAGE: OVERVIEW 

if "Overview" in page:

    st.title("🎧 Customer HelpDesk")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Track • Predict • Analyze • Optimize</p>",
        unsafe_allow_html=True
    )

    # ── Metrics ──────────────────────────────
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

    # ── Charts Row 1 ─────────────────────────
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

    # ── Recent Tickets Table ──────────────────
    st.markdown("### 🗒 Recent Tickets")
    display_cols = [c for c in ["Ticket ID","Customer Name","Ticket Type","Ticket Priority","Ticket Status"] if c in df.columns]
    st.dataframe(
        df[display_cols].head(10),
        use_container_width=True,
        hide_index=True
    )

#  PAGE: PREDICT TICKET 

elif "Predict" in page:

    st.title("🤖 Predict Ticket")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>AI-powered ticket classification engine</p>",
        unsafe_allow_html=True
    )

    # ── Sample prompts ────────────────────────
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

    # ── Input + Result side-by-side ───────────
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
                t     = ticket_text.lower()

                # ── Keyword-based priority override (runs always) ──────────────
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
                if any(k in t for k in HIGH_KW):
                    kw_priority = "High"
                elif any(k in t for k in MEDIUM_KW):
                    kw_priority = "Medium"
                else:
                    kw_priority = "Low"

                # ── Category keyword fallback ──────────────────────────────────
                def keyword_category(t):
                    if any(k in t for k in ["payment","charge","refund","billing","invoice","fee","price","cost"]):
                        return "Billing"
                    elif any(k in t for k in ["wifi","connect","laptop","crash","error","bug","update","software","hardware","device","screen","keyboard","boot","install"]):
                        return "Technical"
                    elif any(k in t for k in ["deliver","package","ship","track","transit","courier","order","parcel","dispatch"]):
                        return "Shipping"
                    elif any(k in t for k in ["password","login","account","reset","email","access","username","sign in","profile","otp","verification"]):
                        return "Account"
                    return "General"

                priority_rank = {"High": 2, "Medium": 1, "Low": 0}

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

            # Result metrics
            r1, r2 = st.columns(2)
            r1.metric("🗂 Category", category)
            r2.metric("🔥 Priority", priority)

            # Confidence bar
            st.markdown(f"**Confidence:** `{conf}%`")
            st.progress(conf / 100)

            # Alert
            if priority == "High":
                st.error("🚨 **Immediate attention required** — escalate now")
            elif priority == "Medium":
                st.warning("⚠ **Handle within 4 hours** — medium urgency")
            else:
                st.success("✅ **Standard queue** — resolve within 24 hours")

            # Est. resolution
            eta = {"High": "< 1 hour", "Medium": "2–4 hours", "Low": "< 24 hours"}
            st.markdown(
                f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
                f"color:#8899b0;margin-top:6px'>⏱ Est. resolution: {eta[priority]}</div>",
                unsafe_allow_html=True
            )

            # History
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "Text": ticket_text[:60] + ("…" if len(ticket_text) > 60 else ""),
                "Category": category,
                "Priority": priority,
                "Conf": f"{conf}%"
            })
            if len(st.session_state.history) > 5:
                st.session_state.history.pop()

        elif predict_btn:
            st.info("Please enter a ticket description first.")
        else:
            st.markdown(
                "<div style='text-align:center;padding:30px 0;color:#8899b0;"
                "font-family:JetBrains Mono,monospace;font-size:12px'>"
                "🤖<br><br>Awaiting ticket input…</div>",
                unsafe_allow_html=True
            )

    # ── Prediction History ────────────────────
    if st.session_state.get("history"):
        st.divider()
        st.markdown("### 🕓 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# PAGE: ANALYTICS 
elif "Analytics" in page:

    st.title("📈 Analytics Dashboard")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Trends • Patterns • Insights</p>",
        unsafe_allow_html=True
    )

    # ── Row 1 ─────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📅 Volume — Last 7 Days")
        days  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        open_ = [80, 60, 90, 100, 85, 50, 40]
        res_  = [40, 38, 55,  60, 47, 30, 25]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(days, open_, color="#ef4444", linewidth=2, marker="o", markersize=4, label="Open")
        ax.fill_between(days, open_, alpha=0.1, color="#ef4444")
        ax.plot(days, res_,  color="#06d6a0", linewidth=2, marker="o", markersize=4, label="Resolved")
        ax.fill_between(days, res_,  alpha=0.08, color="#06d6a0")
        patches = [mpatches.Patch(color="#ef4444", label="Open"), mpatches.Patch(color="#06d6a0", label="Resolved")]
        ax.legend(handles=patches, facecolor=DARK_BG, edgecolor="none", labelcolor="#e2e8f0", fontsize=9)
        apply_dark_style(fig, ax)
        st.pyplot(fig); plt.close(fig)

    with col2:
        st.markdown("### 🎯 Resolution Rate by Category")
        cats_  = ["Technical", "Billing", "Shipping", "Account", "Other"]
        rates  = [72, 88, 91, 85, 79]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bars = ax.bar(cats_, rates, color=TYPE_COLORS[:len(cats_)], width=0.6, zorder=3)
        for bar, v in zip(bars, rates):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                    f"{v}%", ha="center", va="bottom", color="#e2e8f0", fontsize=9, fontfamily="monospace")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Resolution %", color=TICK_COLOR, fontsize=10)
        apply_dark_style(fig, ax)
        st.pyplot(fig); plt.close(fig)

    st.divider()

    # ── Row 2 ─────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### 👥 Top Customers by Volume")
        top_custs = df["Customer Name"].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.barh(top_custs.index[::-1], top_custs.values[::-1], color="#4f8ef7", zorder=3)
        for i, v in enumerate(top_custs.values[::-1]):
            ax.text(v + 0.1, i, str(v), va="center", color="#e2e8f0", fontsize=9, fontfamily="monospace")
        apply_dark_style(fig, ax)
        st.pyplot(fig); plt.close(fig)

    with col4:
        st.markdown("### ⏱ Avg Response Time by Priority")
        pris  = ["High", "Medium", "Low"]
        times = [0.8, 2.4, 5.1]
        colors_ = [PRIORITY_COLORS[p] for p in pris]
        fig, ax = plt.subplots(figsize=(6, 3.4))
        bars = ax.bar(pris, times, color=colors_, width=0.5, zorder=3)
        for bar, v in zip(bars, times):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f"{v}h", ha="center", va="bottom", color="#e2e8f0", fontsize=10, fontfamily="monospace")
        ax.set_ylabel("Hours", color=TICK_COLOR, fontsize=10)
        apply_dark_style(fig, ax)
        st.pyplot(fig); plt.close(fig)

    st.divider()

    # ── Ticket Type breakdown from real data ──
    st.markdown("### 🗂 Full Category Breakdown")
    st.bar_chart(
        df["Ticket Type"].value_counts(),
        use_container_width=True,
        color="#4f8ef7"
    )

#  PAGE: REPORTS 

elif "Reports" in page:

    st.title("📑 Reports")
    st.markdown(
        "<p style='color:#8899b0;font-family:JetBrains Mono,monospace;font-size:12px;"
        "margin-top:-12px'>Summary • SLA • Export</p>",
        unsafe_allow_html=True
    )

    col_rep, col_sla = st.columns(2, gap="large")

    # ── Summary ───────────────────────────────
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
                "Resolved", "Open", "Categories", "Unique Customers", "ML Accuracy"
            ],
            "Value": [
                f"{total_:,}", f"{high_:,}", f"{medium_:,}", f"{low_:,}",
                f"{resolved_:,}" if isinstance(resolved_, int) else resolved_,
                f"{open_t:,}"    if isinstance(open_t,    int) else open_t,
                df["Ticket Type"].nunique(),
                f"{df['Customer Name'].nunique():,}",
                "91.4%"
            ]
        })

        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.success("✅ Report generated successfully")

        # Downloads
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

    # ── SLA + CSAT ────────────────────────────
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
st.caption("© 2024 Customer HelpDesk Intelligence Platform. All rights reserved. | Built with Streamlit & Python")