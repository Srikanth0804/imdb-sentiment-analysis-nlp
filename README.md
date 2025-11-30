# imdb-sentiment-analysis-nlp
IMDB movie review sentiment analysis using RNN, LSTM, GRU and DistilBERT

🎬 **IMDB Sentiment Analysis using Deep Learning & Transformers**

Text Classification with RNN, LSTM, GRU and DistilBERT

📌 **Project Overview**

This project performs sentiment analysis on movie reviews using the IMDB dataset.
Multiple deep learning models were built and compared to understand how different neural architectures perform on text classification tasks.

Classical NLP models were first implemented (RNN / LSTM / GRU), followed by fine-tuning a transformer-based model (DistilBERT) for state-of-the-art results.

The project demonstrates model evolution from basic sequence models to transformer-based architectures.

📊 **Dataset**

IMDB Movie Reviews Dataset

50,000 reviews

Balanced dataset (25,000 positive, 25,000 negative)

Binary labels: Positive (1), Negative (0)

IMDB_Sentiment_Analysis

🛠 **Tech Stack**

Python

TensorFlow / Keras

NumPy, pandas

Matplotlib

scikit-learn

Hugging Face Transformers

DistilBERT

⚙️ **Text Preprocessing**

Implemented text cleaning pipeline:

Lowercasing

HTML tag removal

Special character removal

Whitespace normalization

Tokenization

Padding and truncation

IMDB_Sentiment_Analysis

🧠 **Models Implemented**

1️⃣ Bidirectional RNN

Embedding layer

SimpleRNN

Dense layers

Accuracy ≈ 50% (baseline performance)

2️⃣ Bidirectional LSTM

Learned long-term dependencies

Test Accuracy ≈ 85.1%

3️⃣ Bidirectional GRU

Optimized sequence learning

Test Accuracy ≈ 83.7%

4️⃣ DistilBERT (Transformer Model)

Fine-tuned using Hugging Face

Validation Accuracy ≈ 89.6%

Generated probability scores for predictions

Saved trained model and tokenizer for reuse

IMDB_Sentiment_Analysis

📈 **Results Comparison**

Model	Accuracy

Simple RNN	~50%

LSTM	~85%

GRU	~83%

DistilBERT	~89.6%

🔍 Example Predictions

Example outputs with DistilBERT:

Review: "The movie was absolutely wonderful and emotional"
Prediction: Positive  | Confidence: 0.998

Review: "I hated every second of this boring movie"
Prediction: Negative | Confidence: 0.999

IMDB_Sentiment_Analysis

📁 **Project Structure**

imdb-sentiment-analysis/

│── IMDB_Sentiment_Analysis.ipynb

│── README.md

│── requirements.txt

▶ **How to Run**
Install dependencies:
pip install -r requirements.txt

Run the notebook:
jupyter notebook imdb_sentiment_analysis.ipynb

✅ requirements.txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
transformers
torch

🚀 **Future Enhancements**

Convert into a web app

Add attention visualization

Hyperparameter tuning

Multi-class classification

Model comparison dashboard

👤 **Author**

Srikanth Gunti
📧 Email: srikanthgunti11@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/srikanth-gunti-

⭐ **Support**

If you find this project insightful, feel free to ⭐ star the repository!
