
# Amazon Review Sentiment Analysis

This repository contains a Jupyter Notebook that performs sentiment analysis on Amazon product reviews. The analysis includes text preprocessing, feature extraction, and model training using various machine learning algorithms.

## Project Structure

- **Notebook File**: `Amazon_review_Sentiment_analysis.ipynb`
  - The notebook contains code for loading data, preprocessing text, and training multiple machine learning models.
  
## Key Features

- **Text Preprocessing**: 
  - Tokenization, Lemmatization, and Stopwords removal using NLTK.
  
- **Feature Extraction**:
  - TF-IDF Vectorization to transform text data into numerical format.
  
- **Machine Learning & Deep Learning Models**:
  - Multiple models are used, including Logistic Regression, Random Forest, Decision Trees, SVM, Naive Bayes, and deep learning models like LSTM and CNN.

- **Evaluation**:
  - Model evaluation using accuracy score, classification report, and confusion matrix.

## Data

The notebook uses Amazon product review data. Please ensure that you have the dataset loaded before running the notebook.

## License

This project is licensed under the MIT License.

## Results
◦ Conducted sentiment analysis on product reviews to classify them as positive or negative.
◦ Implemented text pre-processing using NLTK and SpaCy for tokenization, stop words removal, and lemmatization.
◦ Applied Logistic Regression, Random Forest, Decision Tree, Naive Bayes, achieving 85.4% accuracy with Logistic.
◦ Applied DL (RNN, LSTM, Bi-RNN, Bi-LSTM) using Word2Vec with Gensim, achieving 85.6% accuracy with LSTM.
◦ Employed a fine-tuned BERT model transformer from Hugging Face, achieving 90% accuracy in sentiment analysis.
