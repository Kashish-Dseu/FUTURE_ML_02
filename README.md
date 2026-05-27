# 🎧 Customer HelpDesk — Intelligence Platform

An AI-powered customer support dashboard built with **Streamlit**, **NLP**, and **Machine Learning**. Classifies support tickets by category and priority in real time, visualizes trends, and generates exportable reports.

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
├── app.py                        # Main Streamlit application
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

