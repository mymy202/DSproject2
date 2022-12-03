import sys
import pandas as pd
from sqlalchemy import create_engine

'''
This function to load data, clean then merge togethe. I want to clean, deduplicate before merge
imput: message filepath and categories filepath
output: dataset
'''
def load_merge_data(messages_filepath, categories_filepath):
    #messaege
    messages = pd.read_csv(messages_filepath)
    messages.drop_duplicates(keep='first', inplace = True)
    #relace specical charactaire by null
    messages['message'] = messages['message'].str.replace('~[a-z-A-Z- ]','',regex = True)
    
    categories = pd.read_csv(categories_filepath, sep=';', skiprows = 1, header=None )
    numbercol = categories.shape[1]
    categories = categories.join(categories[0].str.split(',', expand=True).rename(
                columns = {0: 'id', 1 : 'related'}
            ))
    categories.drop(columns =[0], inplace = True)

    categories['related'] = categories['related'].str.split('-', expand=True)[1]

    #slipt col
    i = numbercol -1
    while i>0:
        new = categories[i].str.split('-', expand=True)
        categories[new[0][0]] = new[1]
        categories.drop(columns =[i], inplace = True)
        i -= 1

    #change type Convert category values to just numbers 0 or 1.
    categories = categories.astype(int)
    categories.drop_duplicates(keep='first', inplace = True)
    
    # delete value 2 of related
    realated2 = categories['related'] == 2
    categories = categories.drop(categories[realated2].index)

    #delete col child_alone because it is alway 0
    categories = categories.drop(columns=['child_alone'])

    
    df = categories.merge(messages, how = 'inner', on='id')
    return df

# def clean_data(df):
#     pass

'''
This function to save data to sql, and replace if it existe
imput: dataset, filepath
output: 
'''
def save_data(df, database_filename):
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql(database_filename, engine, index=False, if_exists='replace')

'''
This is the main function, it will recall all function above

'''
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_merge_data(messages_filepath, categories_filepath)

#         print('Cleaning data...')
#         df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
