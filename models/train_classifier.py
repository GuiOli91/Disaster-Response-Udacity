import sys
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tempfile import mkdtemp
from shutil import rmtree
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk import download, data
import re
import time
import pickle
from joblib import parallel_backend

# Dowload nltk data
try:
    data.find('tokenizers/punkt')
except LookupError:
    download('punkt')

try:
    data.find('corpora/omw-1.4')
except LookupError:
    download('omw-1.4')

try:
    data.find('corpora/wordnet')
except LookupError:
    download('wordnet')

try:
    data.find('corpora/stopwords')
except LookupError:
    download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath[-19:-3], engine)
    X = df['message']
    Y = df.iloc[,4:]
    return X, Y.to_numpy(), Y.columns.tolist()


def tokenize(text):

    #remove urls
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    #Text Tokenization
    words = word_tokenize(text.lower(), language='english')

    words=[word for word in words if word.isalnum() or len(word) > 1]

    # Tokenize
    words = word_tokenize(text)

    # Remove stop words
    stop = set(stopwords.words('english'))
    words = [word for word in words if word not in stop]

    # Words Lemmatization
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    # Words Stemming

    words = [PorterStemmer().stem(word) for word in words]

    return words


def build_model():
    cachedir = mkdtemp()
    model = Pipeline([('countvect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))],
                    memory=cachedir, verbose = True)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[:,i], y_pred[:,i], zero_division=0))


def save_model(model, model_filepath):
    rmtree(model.memory)
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        with parallel_backend('threading', n_jobs=-3):
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
