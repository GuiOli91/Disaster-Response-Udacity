# Disaster Response Pipeline Project

## Introduction:

Following a disaster (earthquake, flooding, etc...), several messages can be sent directly or indirectly like through social media, and disaster response organizations can have a hard time filtering and classifying the messages to plan their actions more efficiently. This project aims to deliver a web app that will help to classify the disaster messages supplied by [Figure-Eight](https://www.figure-eight.com/). The classification consists to identify 36 labels in several messages like "aid related", "medical help", "missing people" and others to work as an AI filter.

The project is divided into three sections:
* ETL Pipeline
* Machine Learning Pipeline
* Flask Web app

## Files


* Disaster Response Udacity <br/>
| - *app/* <br/>
| | - *templates/* <br>
| | | - **master.html** # main page of web app<br/>
| | | - **go.html** # classification result page of web app<br/>
| | - **run·py** # Python code to run the web app. <br>
| - *data/* <br/>
| | - **disaster_categories.csv** # Categories for each message. <br>
| | - **disaster_messages.csv** # Messages for the AI algorithm. <br>
| | - **DisasterResponse.db** # SQL database to be used by the model <br>
| | - **process_data.py** # Python code for the ETL pipeline. <br>
| - *models/* <br/>
| | - **Data Understanding.ipynb** # Some analysis to understand the data.<br>
| | - **hyperparameters.json** # Json file to support the Gridsearch algorithm. <br>
| | - **ML Pipeline Preparation.ipynb** # Jupyter notebook containing test parts for the Machine learning algorithm. <br>
| | - **report_gridsearch2022*.csv** # Report obtained by the Gridsearchcv.<br>
| | - **train_classifier.py** # Python code to train the model.<br>
| - **.gitattributes** #File to save the model in git lfs <br/>
| - **.gitignore** # Ignores some local files for development<br/>
| - **README.md** # Provides this Introduction.<br/>


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
