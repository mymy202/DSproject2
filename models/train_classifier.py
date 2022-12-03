# import libraries
import sys
import sqlite3
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
import re
import nltk
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import pickle
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

'''
This function to load data
imput: filepath
output: dataset
'''
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    
    # create the test table including project_id as a primary key
    df = pd.read_sql_table(engine.table_names()[0], con=engine)
    X = df['message']
    Y = df.iloc[:,2:36]
    category_names = df.columns[2:36]
    category_names
    return X,Y, category_names

'''
This function to tokenize
imput: text
output: arrange of text
'''
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#     return sent_tokenize(text)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

'''
This function to build model
imput: none
output: dataset
'''
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    
    parameters = {
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4]
    }
  
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return pipeline

    
def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    for col in category_names:
        print("Category: ", col, "\n", classification_report(y_test.loc[:, col].values, y_pred.loc[:, col].values))
        print('Accuracy of' ,col, '%.2f' % (accuracy_score(y_test.loc[:, col].values, y_pred.loc[:, col].values)))

def save_model(model, model_filepath):
#     model_filepath = 'pipeline.pkl'
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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
