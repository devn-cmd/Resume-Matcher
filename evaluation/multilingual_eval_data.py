# evaluation/multilingual_eval_data.py
"""A small, self-contained multilingual evaluation set.

Built from the three role-targeted resumes (AI/ML Engineer, Python Backend,
Data Analyst). Each resume also has a translated copy (DE / FR / ES) so we can
test the core multilingual claim: an ENGLISH job description should still rank
the NON-ENGLISH translation of the correct resume at the top.

PII (email/phone) is intentionally omitted — good practice for a hiring tool,
and the pipeline strips it anyway.

Relevance labels are strict and role-aligned: a JD's relevant resumes are the
matching-role resume in *every* language; all other resumes are non-relevant.
The translated copy of a relevant resume inherits relevance = 1.
"""
import pandas as pd

# ---------------------------------------------------------------- Job descriptions (English)
JD_TEXTS = {
    "jd_ai": (
        "We are hiring an AI/ML Engineer to build end-to-end machine learning "
        "pipelines, RAG systems, and multi-agent LLM workflows. Required: Python, "
        "LangChain, deep learning, NLP, computer vision, vector databases, and "
        "deploying models with Docker."
    ),
    "jd_backend": (
        "Seeking a Python Backend Developer to design scalable REST APIs and "
        "full-stack web platforms. Required: Python, FastAPI, Node.js, React, "
        "PostgreSQL, Docker, and containerized deployment."
    ),
    "jd_data": (
        "Looking for a Data Analyst to build dashboards and statistical models. "
        "Required: SQL, Python, Power BI, Pandas, data cleaning, and data "
        "visualization."
    ),
}

# ---------------------------------------------------------------- Resumes
# English originals (condensed from the uploaded resumes).
_AI_EN = (
    "AI Engineer. Summary: AI/ML Engineer with expertise in end-to-end machine "
    "learning pipelines, RAG architecture, and multi-agent AI workflows using "
    "Python, LangChain, and local LLMs. Builds scalable full-stack AI applications "
    "integrating NLP, computer vision, and API-driven automation. "
    "Technical Skills: Python, JavaScript, SQL, Machine Learning, Deep Learning, "
    "CNN, NLP, RAG, LangChain, LangGraph, LLMs, FAISS, ChromaDB, sentence-transformers, "
    "Computer Vision, TensorFlow, Scikit-learn, Docker, FastAPI, vector databases. "
    "Experience: Data Science Intern at Luminar Technolab. Engineered ML pipelines "
    "and CNN-based computer vision models with Scikit-learn and TensorFlow, "
    "increasing predictive accuracy by 15%. Projects: multi-agent data analysis "
    "pipeline with LangGraph and Llama 3.2; FAISS/Qwen RAG pipeline for document retrieval."
)
_BACKEND_EN = (
    "Python Backend Developer. Summary: Backend Developer with expertise in building "
    "scalable APIs, full-stack web platforms, and automated workflows using Python, "
    "FastAPI, Node.js, and React. Integrates databases and deploys containerized "
    "applications. Technical Skills: Python, JavaScript, HTML, CSS, SQL, React.js, "
    "Node.js, Express.js, FastAPI, Uvicorn, REST APIs, Git, Docker, Docker Compose, "
    "PostgreSQL, OAuth. Experience: RPA Developer at RISS Technologies. Developed "
    "automated workflows for large-scale e-commerce web scraping using Python; "
    "deployed automated email exception handling systems and APIs. Projects: "
    "full-stack booking platform with React, FastAPI, and PostgreSQL with a "
    "JWT-secured admin panel; containerized FastAPI backend via Docker."
)
_DATA_EN = (
    "Data Analyst. Summary: Data Analyst with a strong background in data "
    "visualization, data cleaning pipelines, and statistical analysis. Translates "
    "complex datasets into actionable business insights and interactive dashboards. "
    "Technical Skills: Python, SQL, Power BI, Pandas, Scikit-learn, Jupyter, "
    "predictive modeling, MLflow, PostgreSQL, Git. Experience: Data Science Intern "
    "at Luminar Technolab. Built interactive Power BI dashboards with complex DAX "
    "measures, driving a 20% improvement in executive decision-making; engineered "
    "data pipelines with Pandas and Scikit-learn to clean and process datasets. "
    "Projects: predictive statistical models for employee attrition and job-market "
    "clustering using Python and Pandas; automated data pipeline for cleaning, "
    "anomaly detection, and reporting."
)

# Translated copies (tech terms stay in English, as they do in real resumes).
_AI_DE = (
    "KI-Ingenieur. Zusammenfassung: KI-/ML-Ingenieur mit Fachkenntnissen in "
    "durchgaengigen Machine-Learning-Pipelines, RAG-Architektur und "
    "Multi-Agenten-KI-Workflows mit Python, LangChain und lokalen LLMs. Entwickelt "
    "skalierbare Full-Stack-KI-Anwendungen, die NLP, Computer Vision und "
    "API-gesteuerte Automatisierung integrieren. Technische Faehigkeiten: Python, "
    "JavaScript, SQL, Machine Learning, Deep Learning, CNN, NLP, RAG, LangChain, "
    "LLMs, TensorFlow, Scikit-learn, Computer Vision, Vektordatenbanken, Docker. "
    "Erfahrung: Praktikant im Bereich Data Science bei Luminar Technolab. "
    "Entwicklung von ML-Pipelines und CNN-basierten Computer-Vision-Modellen mit "
    "Scikit-learn und TensorFlow, wodurch die Vorhersagegenauigkeit um 15 Prozent "
    "gesteigert wurde. Projekte: Multi-Agenten-Pipeline zur Datenanalyse mit "
    "LangGraph und Llama 3.2."
)
_BACKEND_FR = (
    "Developpeur backend Python. Resume : Developpeur backend specialise dans la "
    "creation d'API evolutives, de plateformes web full-stack et de flux de travail "
    "automatises avec Python, FastAPI, Node.js et React. Integre des bases de "
    "donnees et deploie des applications conteneurisees. Competences techniques : "
    "Python, JavaScript, HTML, CSS, SQL, React.js, Node.js, Express.js, FastAPI, "
    "Uvicorn, API REST, Git, Docker, Docker Compose, PostgreSQL, OAuth. Experience : "
    "Developpeur RPA chez RISS Technologies. Developpement de flux de travail "
    "automatises pour le web scraping e-commerce a grande echelle avec Python ; "
    "deploiement de systemes automatises de gestion des exceptions par e-mail. "
    "Projets : plateforme de reservation full-stack avec React, FastAPI et "
    "PostgreSQL dotee d'un panneau d'administration securise par JWT."
)
_DATA_ES = (
    "Analista de datos. Resumen: Analista de datos con solida experiencia en "
    "visualizacion de datos, canalizaciones de limpieza de datos y analisis "
    "estadistico. Traduce conjuntos de datos complejos en informacion empresarial "
    "accionable y paneles interactivos. Competencias tecnicas: Python, SQL, Power "
    "BI, Pandas, Scikit-learn, Jupyter, modelado predictivo, MLflow, PostgreSQL, "
    "Git. Experiencia: Pasante de ciencia de datos en Luminar Technolab. Creacion "
    "de paneles interactivos en Power BI con medidas DAX complejas, impulsando una "
    "mejora del 20 por ciento en la toma de decisiones ejecutivas; ingenieria de "
    "canalizaciones de datos con Pandas y Scikit-learn para limpiar y procesar "
    "conjuntos de datos. Proyectos: modelos estadisticos predictivos para la "
    "rotacion de empleados y la agrupacion del mercado laboral con Python y Pandas."
)

RESUMES = {
    "ai_en": _AI_EN,        "ai_de": _AI_DE,
    "backend_en": _BACKEND_EN, "backend_fr": _BACKEND_FR,
    "data_en": _DATA_EN,    "data_es": _DATA_ES,
}

# ---------------------------------------------------------------- Relevance labels
# For each JD, the matching-role resume (in any language) is relevant = 1.
_RELEVANT = {
    "jd_ai":      {"ai_en", "ai_de"},
    "jd_backend": {"backend_en", "backend_fr"},
    "jd_data":    {"data_en", "data_es"},
}


def build_labels() -> pd.DataFrame:
    rows = []
    for jd_id in JD_TEXTS:
        for resume_id in RESUMES:
            rows.append({
                "jd_id": jd_id,
                "resume_id": resume_id,
                "relevant": int(resume_id in _RELEVANT[jd_id]),
            })
    return pd.DataFrame(rows)


def load_dataset():
    """Return (jd_texts, resumes, labels) ready for evaluate()."""
    return JD_TEXTS, RESUMES, build_labels()