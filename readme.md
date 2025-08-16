# VAD-Aware Emotion and Sentiment Classification

## Overview

This project implements a **Valence-Arousal-Dominance (VAD) aware** model for **emotion and sentiment classification**. Unlike traditional approaches that rely solely on text features, this model incorporates **VAD embeddings** to better capture the emotional nuances of language, improving classification performance on both **emotion recognition** and **sentiment analysis** tasks.

> **Note:** This project was part of the **CS626 course** and was completed as a joint effort by **Aryan Khilwani, Rajshekar K, Sejal Kadamdhad, and Aditya Hol**.

## Features

- Emotion classification (e.g., joy, sadness, anger, fear, etc.)
- Sentiment classification (positive, negative, neutral)
- Integration of **VAD embeddings** to enhance emotional understanding
- Compatible with multiple datasets for benchmarking
- Easy-to-use training and evaluation scripts

## Requirements

- Python 3.8+
- PyTorch >= 2.0
- Transformers (HuggingFace)
- scikit-learn
- pandas, numpy
- tqdm
- matplotlib / seaborn (for visualization)

## Datasets

- **GoEmotions**: Fine-grained emotion dataset for English text  
  [GitHub Link](https://github.com/google-research/google-research/tree/master/goemotions)
  - Used for **emotion classification**
- **SST-2 (Stanford Sentiment Treebank)**: Sentiment analysis dataset  
  [Dataset Link](https://nlp.stanford.edu/sentiment/index.html)
  - Used for **sentiment classification**

> The datasets should be preprocessed to include: text inputs, labels, and optional VAD scores (can be mapped from lexicons if not available).

## Model

- **Text Encoder:** BERT (pre-trained)
- **VAD Embeddings:** Valence, Arousal, Dominance
- **Fusion Layer:** Combines BERT features with VAD embeddings
- **Classifier Heads:** Separate heads for emotion and sentiment prediction

## This Project was made as a
