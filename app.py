from distutils.log import debug
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import re
import pickle
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import Word2Vec


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def preprocess_text(sent):
    #lowercase
    sentences = sent.lower()
    #remove other character except alphabet
    sentences = re.sub(r'[^a-zA-Z]', ' ', sentences)
    # Single character removal
    sentences = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentences)
    # Removing multiple spaces
    sentences = re.sub(r'\s+', ' ', sentences)
    return sentences

# nltk.download('punkt')
# nltk.download('stopwords')
# listStopword = set(stopwords.words('indonesian'))
# listStopword.update(set(stopwords.words('english')))

# def stopwords_removal(sent):
#     sents = [s for s in sent.split(" ") if s not in listStopword]
#     return " ".join(sents)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    model = tf.keras.models.load_model('lstm_model_10fold.h5')

    if request.method == 'POST':
        text = request.form.get('text')
        data = [preprocess_text(text)]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(data)
        text_seq = tokenizer.texts_to_sequences(data)
        text_pad = pad_sequences(text_seq, padding='post', truncating= 'post', maxlen=2425)

        predict_val = model.predict(text_pad)
        
        if (predict_val >= 0.50):
            prediction = "Informasi Ini benar!"
        else:
            prediction = "Informasi Ini Hoaks!"
    
    return render_template('results.html', prediction = prediction, pred_val=predict_val, in_text = text, prep_text = data)

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)