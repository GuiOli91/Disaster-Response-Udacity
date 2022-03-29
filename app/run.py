import json
import plotly.express as px
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Treemap

import pickle
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Returns the master.html concatenate with the plotly graphs.

    Returns:

    string : Renderization of the master.html with the graphs.
    """

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Top 10 labels

    countcategory = df.iloc[:,4:].sum(axis=0)
    countcategory = countcategory.sort_values(ascending=False)

    #Quantity of labels per message

    countlabels = df.iloc[:,4:].sum(axis = 1)
    countlabels = countlabels.value_counts()
    labels = [str(label) +' labels' for label in countlabels.index]
    values = countlabels.values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=countcategory[:10].index,
                    y=countcategory[:10].values

                )

            ],

            'layout': {
                'title': 'Top 10 Label\'s Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels"
                }
            }
        },
        {
            'data': [
                Treemap(
                    labels=labels,
                    parents=[""]*len(labels),
                    values =  values,

                )

            ],

            'layout': {
                'title': 'Label\'s Quantity per message',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Labels"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graph_json = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graph_json)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Returns the go.html concatenate with the query.

    Returns:

    string : Renderization of the go.html with the graphs.
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
