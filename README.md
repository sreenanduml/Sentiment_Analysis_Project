SENTIMENT ANALYSIS APP

Sentiment Analysis is a Natural Language Processing (NLP) technique used to identify and classify emotions expressed in text. It is widely applied in areas such as social media monitoring, product review analysis, customer feedback evaluation, and opinion mining.                                
This project implements a Sentiment Analysis Web Application using Machine Learning and Flask. The application accepts textual input from users and predicts the sentiment as Positive, Negative, or Neutral. The model is trained on a labeled dataset containing tweets/reviews and is capable of performing both real-time predictions and batch predictions on CSV files.
The goal of this project is to demonstrate the end-to-end pipeline of a machine learning application — from data preprocessing and model training to deployment using a web framework.
sentiment_app/


├── data/

   ├── test (1).csv

├── model/

   ├── sentiment_model.pkl
   
   └── vectorizer.pkl

├── templates/

   └── index.html

├── train.py

├── app.py

├── predict.py

├── requirements.txt

└── README.md

This project is a Sentiment Analysis application built using Python, Machine Learning, and Flask.
It predicts whether a given text expresses a Positive, Negative, or Neutral sentiment.

Features:

Trained ML model using TF-IDF + Logistic Regression and used Flask-based web application for real-time predictions
