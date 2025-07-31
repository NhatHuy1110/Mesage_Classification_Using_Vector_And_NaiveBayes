# 📩 Message Classification Using Vector Database and Naive Bayes.

## 🎯 Project Purpose

The project aims to classify messages into groups such as "spam" or "ham" (valid messages) using two different approaches:
- Classical machine learning model: **Naive Bayes**.
- Semantic search system based on **Vector Database (FAISS)** combined with language model from Hugging Face.
-> The objective is to compare the performance of traditional and state-of-the-art methods in short text classification.
  
## 📄 Project Overview

### 1. Naive Bayes Classification
- Text preprocessing: case conversion, punctuation, word segmentation, stopwords.
- Vectorization of data using **Bag-of-Words** method (manually).
- Classification using **Multinomial Naive Bayes** from the `scikit-learn` library.
- Accuracy assessment using `accuracy_score`.

### 2. Vector Database with Sentence Embedding (FAISS + Transformer)
- Using the `intfloat/multilingual-e5-base` model from Hugging Face to create **sentence embeddings**.
- Data is encoded into vectors and stored using **FAISS** – an optimized library for approximate nearest neighbor search.
- Querying by converting input sentences into vectors and finding the closest vectors in the database.
- Classification based on the labels of the closest similar vectors.

## 📂 Dataset

- **File name**: `2cls_spam_text_cls.csv`
- **Source**: Download directly from Google Drive (`gdown`)
- **Structure**: Includes two main columns:
  + `Message`: message content (text string)
  + `Category`: classification label (`spam` or `ham`)
- **Objective**: Classify input text content into two classes based on training information.

## ⚙️ Key Technologies & Libraries

- `scikit-learn`: train Naive Bayes models and evaluate performance.

- `nltk`: natural language processing (word segmentation, stopwords, etc.).

- `transformers`: take pretrained language models.

- `faiss`: build vector databases for fast searching.

- `torch`: process tensors and embeddings.

- `tqdm`, `pandas`, `numpy`: process data and visualize progress.

## 📌 Author & Copyright

Academic project to compare the performance of text classification methods. All data and models are public or open source.
