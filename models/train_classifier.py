import sys
import argparse
import os.path
import json
import functools
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
from datetime import datetime


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
from sklearn.metrics import classification_report, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt

# Create custom options to run the code.

parser = argparse.ArgumentParser(description = "The script trains a \
classifier model for the Disaster Response - Udacity project.")

parser.add_argument("database", help="The path to the SQL database that will be\
 used to train the model.")

parser.add_argument("model", help="The file where the the trained model will be\
 saved.")

parser.add_argument("-v", "--verbose", help=" Will output the evaluation",
 action = "store_true")

parser.add_argument("-g", "--gridsearch", help = "Json file that will be used \
to do a GridSearchCV to search for hyperparameters")

parser.add_argument("-r", "--report", help = "Saves a report from the training",
action = "store_true")


args = parser.parse_args()


def load_data(database_filepath):
    """
    Loads the data from a SQL database and returns the values for the model.

    Parameters:
    database_filepath (string): Path to the SQL Database

    Returns:
    pandas.Series:  Serie with the inputs for the trains
    numpy.ndarray:  Matrix with the labels for each inputs
    list:           List with name for each labels
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath[-19:-3], engine)
    # NOTE: remove iloc on lines
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y.to_numpy(), Y.columns.tolist()


def tokenize(text):
    """
    Returns tokens from messages.

    Receives unstructured data (messages) and do some taks:
        - Removes urls starting with http.
        - Separete each message by word.
        - Removes stop words.
        - Lemmatize the words.
        - Stem the words.

    Parameters:
    text (list):    Array like of strings.

    Returns:
    list:           2d List of tokens.
    """

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


def build_model(verbose = True):
    """
    Returns a pre-defined model to be trained by the algorithm.

    Parameters:
    verbose (boolean):  Sets the verbosity of the model.

    Returns:
    sklearn.pipeline.Pipeline:  A model defined in a Pipeline.
    """
    cachedir = mkdtemp()
    model = Pipeline([('countvect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))],
                    memory=cachedir, verbose = verbose)
    return model


def evaluate_model(model, X_test, Y_test, category_names, verbose):
    """
    Prints the confusion matrix per label

    Parameters:
    model (sklearn.pipeline.Pipeline): Trained model
    X_test (pandas.series): Serie with messages.
    Y_test (numpy.ndarray): Matrix with labels for each input
    category_names (list): List with name for each labels
    verbose (boolean): If True outputs the results on the terminal

    Returns:
    sklearn.pipeline.Pipeline:  A model defined in a Pipeline.
    """
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        if verbose:
            print(category_names[i])
            print(classification_report(Y_test[:,i], y_pred[:,i], zero_division=0))


def save_model(model, model_filepath):
    """
    Saves the model in a binary format on the received path.

    Parameters:
    model (sklearn.pipeline.Pipeline): Trained model to be saved.
    model_filepath (string): Path to where the model will be saved
    """
    rmtree(model.memory)
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():

    print('Loading data...\n    DATABASE: {}'.format(args.database))
    X, Y, category_names = load_data(args.database)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


    if args.gridsearch:
        if os.path.exists(args.gridsearch):
            try:
                with open(args.gridsearch, 'r') as file:
                    gridparameter = json.load(file)
            except Exception as e:
                raise TypeError("Only json files are allowed")

            print('Building a model...')
            model = build_model(verbose=False)
            print('Searching for hyperparameters...')
            # f1_score_macro = functools.partial(f1_score, average='macro', zero_division=0)
            cv = GridSearchCV(model, param_grid = gridparameter, scoring = 'f1_samples', verbose = 3)
            cv.fit(X_train, Y_train)
            model = cv.best_estimator_

        else:
            print(f"The file {args.gridsearch} doesn't exist.")
            return
    else:

        print('Building model...')
        model = build_model()

        print('Training model...')
        with parallel_backend('threading', n_jobs=-3):
            model.fit(X_train, Y_train)


    print('Evaluating model...')

    if args.report:
        path = os.path.split(os.path.abspath(__file__))[0]
        date = datetime.today().strftime('%Y%m%d_%H%M%S')
        if args.gridsearch:
            results = pd.DataFrame(cv.cv_results_)
            file = os.path.join(path, "report_gridsearch" + date + ".csv")
            results.to_csv(file)

    evaluate_model(model, X_test, Y_test, category_names, verbose=args.verbose)

    print('Saving model...\n    MODEL: {}'.format(args.model))
    save_model(model, args.model)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
