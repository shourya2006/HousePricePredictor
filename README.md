# Project 9: Intelligent Property Price Prediction & Agentic Real Estate Advisory

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## From Predictive Analytics to Market Insights

### Project Overview

This project involves the design and implementation of an analytics system that predicts property prices and evolves into an agentic AI real estate advisory assistant.

- **Milestone 1:** The system applies classical machine learning techniques to historical listing data and location attributes to predict property values and analyze market drivers.
- **Milestone 2:** The same system is extended into an agent-based AI application that autonomously reasons about property characteristics, retrieves market insights, and generates investment recommendations.

The project emphasizes real-world real estate decision-making, progressive development from predictive analytics to agentic AI workflows, and public deployment.

---

### Constraints & Requirements

- **Development Environment:** Jupyter Notebooks, Streamlit
- **ML Pipeline:** Scikit-Learn (RandomForestRegressor, LinearRegression, StandardScaler)
- **Data Source:** Ames Housing Dataset

---

### Technology Stack

| Component               | Technology                                               |
| :---------------------- | :------------------------------------------------------- |
| **Machine Learning**    | Random Forest Regressor, Linear Regression, Scikit-Learn |
| **Data Processing**     | Pandas, Numpy, Label Encoding                            |
| **UI Framework**        | Streamlit, Custom HTML/CSS                               |
| **EDA & Visualization** | Matplotlib, Seaborn, Plotly                              |

---

### Milestones & Deliverables

#### Milestone 1: ML-Based Price Prediction

**Objective:**
Identify the core drivers of house prices and develop an accurate predictive model using a custom machine learning pipeline.

**Key Deliverables:**

- **Exploratory Data Analysis (EDA):** Comprehensive correlation heatmaps and distribution graphs.
- **Feature Engineering:** Handling of null values and categorical label encoding.
- **Robust ML Pipeline:** A seamless Scikit-Learn pipeline integrating data scaling and predictive regression.
- **Model Evaluation:** Performance metrics computing R-Squared ($R^2$) scores to measure model accuracy.

#### Milestone 2: Interactive Estimator Application

**Objective:** Deploy the predictive model into a clean, minimal user interface allowing for real-time inference.

**Key Deliverables:**

- Working local web application with an intuitive UI (Streamlit).
- Dynamic categorical mapping system (translating natural String inputs to ML-expected LabelEncodings).
- Organized 3-column component structure.

---

### Team Members & Contributions

This project was developed collaboratively by the following team members:

| Team Member          | Tasks & Responsibilities                                                           |
| :------------------- | :--------------------------------------------------------------------------------- |
| **Shourya Bafna**    | Data Exploration & Preprocessing (EDA, Data Cleaning, Heatmaps)                    |
| **Aditya Bharadwaj** | Feature Engineering & Data Transformation (Null handling, Label Encoding)          |
| **Daksh Batra**      | Model Selection & Pipeline Architecture (Scikit-Learn Regression, Feature Scaling) |
| **Om Yadav**         | UI/UX Engineering & Web App Deployment (Streamlit layout, Frontend mappings)       |

---

### Installation & Setup

Follow these instructions to run both the analytical notebooks and the frontend machine learning application locally on your machine.

**1. Clone the repository**

```bash
git clone https://github.com/shourya2006/HousePricePredictor.git
cd HousePricePredictor
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Run the Streamlit Web Application**
To interact with the frontend estimator UI:

```bash
streamlit run main.py
```

**4. View the Analytical Notebook**
To explore the core ML pipeline, feature engineering, and data processing steps behind the scenes:

```bash
jupyter notebook app.ipynb
```

> **Note:** Ensure you have the datasets present in the root directory before running the Jupyter Notebook cells.
