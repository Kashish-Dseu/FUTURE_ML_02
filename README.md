# 🎧 Customer HelpDesk — Intelligence Platform

An AI-powered customer support dashboard built with **Streamlit**, **NLP**, and **Machine Learning**. Classifies support tickets by category and priority in real time, visualizes trends, and generates exportable reports.

---
## 🧭 Project Overview

Customer support teams receive hundreds of tickets daily — from billing disputes and delivery delays to account lockouts and technical crashes. Manually reading, categorizing, and prioritizing each one wastes agent time and slows down resolution.

**Customer HelpDesk** solves this by combining a trained ML classifier with a keyword-aware NLP pipeline to:

- **Automatically classify** every incoming ticket into one of 5 categories: `Technical`, `Billing`, `Shipping`, `Account`, `General`
- **Assign priority** (`High / Medium / Low`) based on urgency signals in the text
- **Explain decisions** by surfacing the exact keywords that drove each prediction
- **Track patterns** across all predictions made in a session with live analytics
- **Report on SLA compliance** and customer satisfaction scores

### Who is this for?

| Role | How they use it |
|---|---|
| **Support agents** | Paste a ticket description → get instant category + priority + ETA |
| **Team leads** | Monitor the Analytics page for session-level trends and escalation patterns |
| **Data scientists** | Swap in their own `.pkl` models; compare ML vs keyword accuracy in the sidebar |
| **Managers** | Pull the Reports page for SLA stats, CSAT scores, and exportable summaries |

---

## 📸 Features

| Feature | Description |
|---|---|
| 🏠 Overview | Live KPI metrics, ticket distribution charts, recent ticket log |
| 🤖 Predict Ticket | AI-powered category & priority classification with keyword detection |
| 📈 Analytics | Session-level charts — timeline, category breakdown, priority matrix, model accuracy |
| 📑 Reports | SLA performance, CSAT scores, summary stats, CSV export |

---

## 🗂 Project Structure

```
.
├── Task2.py                        # Main Streamlit application
├── customer_support_tickets.csv  # Dataset (required)
├── category_model.pkl            # Trained category classifier (optional)
├── priority_model.pkl            # Trained priority classifier (optional)
├── vectorizer.pkl                # TF-IDF vectorizer (optional)
└── README.md
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI framework | [Streamlit](https://streamlit.io) |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (TF-IDF + classifier) |
| NLP | NLTK (stopword removal, text cleaning) |
| Visualisation | Matplotlib |
| Fonts | Space Grotesk, JetBrains Mono (Google Fonts) |

---
## 🔄 Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TICKET SUBMITTED                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — TEXT CLEANING  (clean_text)                           │
│     Lowercase  →  Remove punctuation  →  Strip stopwords        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               │  ML models present?           │
              YES                              NO
               │                               │
               ▼                               ▼
  ┌────────────────────────┐     ┌─────────────────────────┐
  │  TF-IDF Vectorizer     │     │  KEYWORD FALLBACK        │
  │  → Category Classifier │     │  keyword_category()      │
  │  → Priority Classifier │     │  + HIGH_KW / MEDIUM_KW   │
  └────────────┬───────────┘     └────────────┬────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — KEYWORD OVERRIDE  (detect_keywords)                   │
│     Scan raw text for HIGH_KW / MEDIUM_KW / CATEGORY_KW         │
│     If keyword urgency > ML urgency  →  keyword result wins     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — RESULT ASSEMBLED                                      │
│     Category   · Priority   · Confidence %                      │
│     Est. resolution time   · Colour-coded keyword tags          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — SESSION HISTORY  (st.session_state)                   │
│     Every prediction stored: text snippet, category, priority,  │
│     confidence score, matched keywords                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌────────────────────────┐     ┌─────────────────────────┐
  │  ANALYTICS PAGE        │     │  REPORTS PAGE            │
  │  • Category bar chart  │     │  • SLA compliance bars   │
  │  • Priority pie chart  │     │  • CSAT scores           │
  │  • Timeline scatter    │     │  • Summary metrics table │
  │  • F1 accuracy bars    │     │  • CSV export            │
  │  • Category×Priority   │     └─────────────────────────┘
  │    grouped bar chart   │
  └────────────────────────┘
```

