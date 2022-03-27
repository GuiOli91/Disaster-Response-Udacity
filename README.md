# Disaster Response Pipeline Project

## Introduction:

This project aims to deliver a web app that will help to classify the disaster messages supplied by [Figure-Eight](https://www.figure-eight.com/). The classification consists of the identification of 36 labels in several messages like **aid related**, **medical help**, **missing people** etc.

The project is divided into three sections:
* ETL Pipeline
* Machine Learning Pipeline
* Flask Web app

## Technologies

* **scikit-learn** - 1.0.2
* **plotly** - 5.6.0
* **flask** - 2.0.2
* **python** - 3.9.7
* **nltk** - 3.7

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`. The *ML  pipeline* can be run with some options:
        - **-v** or **--verbose**: Will output the Evaluation for the classifier.
        - **-g GRIDSEARCH**  or **--gridsearch GRIDSEARCH**: Will receive a Json file containing parameters to search for hyperparameters.
        - **-r** or **--report**: Saves a report for the gridsearch.


2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
