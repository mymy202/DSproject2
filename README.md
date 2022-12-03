# Disaster Response Pipeline Project

### Project Overview
This project is  to analyze disaster data from Appen to build a model for an API that classifies disaster messages.

### file included
- App: have api and web coding
- data: have file coding to load and clean data and file data
- models: have coding and file model
- readme: have all information about project


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/etl_pipeline.py data/disaster_messages.csv data/disaster_categories.csv data/InsertDatabaseName.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/pipeline.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
