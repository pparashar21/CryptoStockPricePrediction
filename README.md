# CryptoStockPricePrediction

A Real-Time Cryptocurrency Trends Analysis Platform using sentiment and persuasion analysis from news and social media to predict crypto price movements. Built with machine learning, NLP, and real-time data processing, the platform helps investors and analysts make informed decisions in the highly volatile crypto market.

---

## Overview

This project explores how **sentiment** and **persuasive language** in public news and social media influence cryptocurrency price movements. Inspired by real-world phenomena (like Elon Musk’s tweets impacting Dogecoin), our platform performs dual-layer NLP analysis and feeds those features into machine learning models for price movement prediction.

---

## Key Features

- **Multi-source Data Integration**  
  Scrapes and ingests data from Google News, Yahoo Finance, Twitter (via HuggingFace), and News APIs.

- **Sentiment & Persuasion Analysis**  
  - Sentiment via FinBERT + TextBlob and GPT-4o-mini  
  - Persuasion detection using 23 SemEval techniques via LLM-based few-shot prompting

- **Price Prediction Models**  
  Logistic Regression, SVM, Random Forest, MLP, XGBoost, LSTM

- **Interactive GUI**  
  Built with Tkinter for live visualization and interaction

- **Feature Engineering**  
  Lag features, moving averages, volatility, sentiment polarity, subjectivity, persuasion scores

---

## Results

| Model                  | Accuracy | ROC-AUC |
|-----------------------|----------|---------|
| Random Forest          | 0.64     | 0.65    |
| Multi-layer Perceptron | 0.59     | 0.62    |
| SVM                    | 0.57     | 0.41    |
| XGBoost                | 0.54     | 0.56    |
| LSTM                   | 0.53     | 0.53    |

**Random Forest** emerged as the best-performing model, driven by:
- Sentiment polarity and subjectivity
- Persuasion scores (SemEval-based)
- Lagged and rolling statistical features

### Comparison with Prior Research

Compared to notable prior work:
- **Loginova et al. (2021, Springer)** achieved a **ROC-AUC of 0.58** using MLP  
- **MDPI (2021)** reported an **accuracy of 0.61** using conventional sentiment analysis techniques

 **Our project improved upon this by:**
- Achieving **12% higher ROC-AUC** (0.65 vs 0.58)
- Delivering **5% better accuracy** (0.64 vs 0.61)
- Introducing **persuasion detection** as a novel NLP signal for financial forecasting — an innovation not seen in earlier works

  This indicates a strong contribution in enhancing cryptocurrency prediction using modern LLM-based sentiment/persuasion modeling and integrated feature engineering.
---

## Tech Stack

- **Language**: Python  
- **NLP**: FinBERT, TextBlob, GPT-4o-mini, NLTK  
- **ML/DL**: Scikit-learn, TensorFlow, Keras  
- **Visualization**: Matplotlib, Seaborn, WordCloud  
- **Data Collection**: Selenium, PRAW, NewsAPI  
- **GUI**: Tkinter  
- **OS Support**: macOS (AppKit used for NSWorkspace)

---

## Future Enhancements

- Real-time streaming via CoinGecko API  
- Signal-based trading alert system  
- Image-based visual sentiment analysis  
- Web dashboard using Django or Streamlit  
- High-frequency forecasting with LSTM/Transformers

---

## Project Report

A **detailed write-up** of this project — including background, methodology, analysis, model comparisons, and future work — is available in the repository.

[Click here to view the full project report](./Final_Report.pdf)

We encourage readers to review the report for a deep dive into:
- Theoretical motivation
- NLP prompt engineering for GPT-4o
- EDA and data pipeline details
- Model evaluation and tuning
- Innovation in persuasion analysis and feature engineering
