# Project 9: Intelligent Property Price Prediction & Agentic Real Estate Advisory

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## From Predictive Analytics to Autonomous Investment Advice

### Project Overview

This project is a two-part educational journey. It begins as a classical machine learning application built to predict property prices, and evolves into a cutting-edge **Agentic AI Real Estate Advisory Assistant**.

- **Milestone 1:** The system applies classical machine learning pipelines to historical listing data and location attributes to predict property values and analyze market drivers.
- **Milestone 2:** The system is supercharged into an agent-based AI application (using LangGraph and OpenAI) that autonomously reasons about property characteristics, retrieves market insights via RAG, and generates downloadable investment reports.

---

### Technology Stack

| Component                     | Technology                                               |
| :---------------------------- | :------------------------------------------------------- |
| **Machine Learning**    | Random Forest Regressor, Linear Regression, Scikit-Learn |
| **Agentic Framework**   | LangChain, LangGraph, OpenAI (GPT-4o-mini)               |
| **Vector Database**     | ChromaDB, OpenAI Embeddings                              |
| **UI Framework**        | Streamlit, Custom HTML/CSS                               |
| **PDF Generation**      | ReportLab                                                |
| **EDA & Visualization** | Pandas, Matplotlib, Seaborn                              |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Price Predictor

**Objective:** Identify the core drivers of house prices and develop an accurate predictive model using a custom machine learning pipeline, deployed to a clean UI.

**Key Deliverables:**

- **Exploratory Data Analysis (EDA):** Comprehensive correlation heatmaps and distribution graphs.
- **Robust ML Pipeline:** A seamless Scikit-Learn pipeline integrating data scaling and predictive regression.
- **Interactive UI:** A Streamlit dashboard allowing users to input property features (like square footage, year built, garage size) to get real-time price inferences from the trained `rf_model.jb`.

#### Milestone 2: Agentic AI Real Estate Advisor

**Objective:** Transform the basic predictive model into an autonomous AI agent capable of dynamic reasoning, market research, and report generation.

**Key Deliverables:**

- **RAG Knowledge Base:** Integrated ChromaDB to store and retrieve real estate market trends, investment strategies, and localized housing reports.
- **Auto-Routing ReAct Agent:** Built a conversational LangGraph agent that intelligently routes human queries to the correct backend tools (Price Predictor, Market Insights RAG, or Comparable Sales Finder) without manual intervention.
- **Agentic Tooling:** Custom tools enabling the LLM to execute python functions and query the `AmesHousing` dataset for similar properties using weighted Euclidean distance.
- **Professional Report Generation:** A standalone module that captures the agent's context and generates a polished, downloadable PDF Advisory Report using `reportlab`.

---

### Team Members & Contributions

This project was developed collaboratively by the following team members:

| Team Member                | Tasks & Responsibilities                                                                 |
| :------------------------- | :--------------------------------------------------------------------------------------- |
| **Shourya Bafna**    | RAG Setup & conversational Chat Agent architecture (`rag_setup.py`, `chat_agent.py`) |
| **Aditya Bharadwaj** | Backend AI Tooling & dataset utility functions (`tools.py`)                            |
| **Daksh Batra**      | LangGraph workflow orchestration & state management (`graph.py`, `state.py`)         |
| **Om Yadav**         | Frontend UI/UX Engineering & Web App Deployment (Streamlit layout)                       |

---

### Installation & Setup

Follow these instructions to run the Agentic Streamlit application locally.

**1. Clone the repository**

```bash
git clone https://github.com/shourya2006/HousePricePredictor.git
cd HousePricePredictor
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory and add your OpenAI API key (required for the AI Agent and embeddings):

```env
OPENAI_API_KEY=your-api-key-here
```

**4. Initialize the Vector Database (One-time setup)**
Before using the AI Advisor, you must build the local ChromaDB vector store:

```bash
python agent/rag_setup.py
```

*(This will embed the `knowledge_base` and create a `chroma_db` folder).*

**5. Run the Streamlit Web Application**
Launch the two-tab dashboard (Price Estimator & AI Advisor):

```bash
streamlit run main.py
```
