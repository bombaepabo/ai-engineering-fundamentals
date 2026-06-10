# AI & Deep Learning: A 5-Project Roadmap 🚀🎓

Welcome to the **AI & Deep Learning Learning Curriculum**. This repository contains a structured, hands-on roadmap designed to take a developer from classical machine learning baselines to state-of-the-art Generative AI pipelines and production-grade AI engineering. 

Each project builds on the skills learned in the previous one, transitioning from simple algorithms to neural networks, Transformers, and LLM integrations.

---

## 🗺️ The 5-Project Learning Path

```text
  Phase 1: Tabular ML       Phase 2: Deep Learning    Phase 3: Transformers
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│       Project 1       │  │       Project 2       │  │       Project 3       │
│  Telco Customer Churn │──│   SVHN Digit CNN      │──│ Sentiment Classifier  │
│  (sklearn Classifier) │  │  (PyTorch CV Model)   │  │ (DistilBERT NLP Model)│
└───────────────────────┘  └───────────────────────┘  └───────────────────────┘
                                                                  │
                                                                  ▼
  Phase 5: AI Production    Phase 4: Vector RAG       Vector RAG Integration
┌───────────────────────┐  ┌───────────────────────┐
│       Project 5       │  │       Project 4       │
│ AI Support Ticket App │◄─│  Gemini RAG Chatbot   │
│ (FastAPI & pgvector)  │  │ (ChromaDB & Gemini)   │
└───────────────────────┘  └───────────────────────┘
```

---

## 📚 Project Directory

### [Project 1: Churn Prediction (Tabular ML)](./project_1_churn_prediction/)
* **Focus**: Tabular Data, Feature Engineering, and Classical Supervised Learning.
* **Problem**: Predict subscription churn from customer billing and usage metrics.
* **Core Tech**: `scikit-learn`, `pandas`, `joblib`.
* **Key Concepts**:
  * Cleaning dirty data, handling missing variables, and encoding categories (One-Hot Encoding).
  * Comparing baseline algorithms (Logistic Regression vs. Random Forest).
  * Hyperparameter tuning (`GridSearchCV`) and selecting models based on custom metrics like Recall and F1-score rather than basic accuracy.

### [Project 2: SVHN Digit Classifier (Computer Vision)](./project_2_svhn_digit_classifier/)
* **Focus**: Neural Networks, Image Operations, and Deep Learning.
* **Problem**: Classify single digits from Street View House Numbers (SVHN) images.
* **Core Tech**: `PyTorch`, `torchvision`, `numpy`.
* **Key Concepts**:
  * Constructing Convolutional Neural Network (CNN) layers (`Conv2d`, `ReLU`, `MaxPool2d`).
  * Creating a custom training and evaluation validation loop.
  * Combating neural network overfitting using `Dropout` layers.

### [Project 3: Sentiment Classifier (Transformers & NLP)](./project_3_transformer_sentiment_classifier/)
* **Focus**: Natural Language Processing (NLP) and Transfer Learning.
* **Problem**: Predict sentiment classes (positive, neutral, negative) for raw review text.
* **Core Tech**: `Hugging Face Transformers` (DistilBERT), `PyTorch`, `datasets`.
* **Key Concepts**:
  * Tokenizing raw text into token IDs for pretrained architectures.
  * Comparing classical TF-IDF baselines with deep transformer models.
  * Fine-tuning a pre-trained **DistilBERT** sequence classification model using the Hugging Face `Trainer` API.

### [Project 4: Gemini RAG Chatbot (Vector Database & LLMs)](./project_4_gemini_rag_chatbot/)
* **Focus**: Generative AI, Retrieval-Augmented Generation (RAG).
* **Problem**: Chat with local PDF/text documents and guarantee factual responses.
* **Core Tech**: `ChromaDB` (vector store), `google-genai` SDK (`gemini-2.5-flash-lite` & `text-embedding-004`).
* **Key Concepts**:
  * Chunking long documents and generating vector embeddings.
  * Indexing embeddings inside a local vector database.
  * Query similarity search and injecting matched text into LLM prompts (grounding).

### [Project 5: Support Ticket Intelligence (Production AI Engineering)](./project_5_support_ticket_intelligence/)
* **Focus**: Full Stack AI Integration, API design, Observability, and Hybrid Architectures.
* **Problem**: Build a complete, production-grade automated support ticket agent backend.
* **Core Tech**: `FastAPI`, `SQLAlchemy`, `pgvector` (PostgreSQL), `alembic`, `google-genai`, `pytest`, `Docker`.
* **Key Concepts**:
  * **Hybrid AI Architecture**: Running high-speed, local ML models (from Project 1) for routing, alongside LLMs (from Project 4) for summary and reply drafts.
  * **Relational Vector Database**: Indexing and searching vectors natively inside PostgreSQL using `pgvector`.
  * **Asynchronous Lifecycles**: Running heavy processing tasks in background threads to keep HTTP response times fast.
  * **Stateful Chat Memory**: Utilizing database tables to maintain conversational context.
  * **Production Observability**: Generating unique Request IDs and formatting logging streams into JSON structured outputs.
  * **Containerization**: Packaging the stack into a Docker recipe.

---

## 🛠️ Workspace Prerequisites

To run these projects, you need to have the following tools installed on your host system:

1. **Python 3.11+**
2. **`uv`** (Astral's fast Python package manager)
3. **Docker & Docker Compose** (For running PostgreSQL with the pgvector extension)
