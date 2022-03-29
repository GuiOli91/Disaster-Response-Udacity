import sys
import pandas as pd
import time
from sqlalchemy import create_engine

startTime = time.time()


def load_data(messages_filepath, categories_filepath):
    """
    Load csv data from a received paths.

    Parameters:
    messages_filepath (string): Path to the messages file.
    categories_filepath (string): Path to the categories file.

    Returns:
    pandas.Dataframe:  Merged Dataframe.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the datasets
    return messages.merge(categories, left_on = 'id', right_on = 'id')

def clean_data(df):
    """
    Cleans a Dataframe.

        - Split the categories.
        - Transfom the string from the labels into 0 or 1
        - Drop duplicate rows.
        - Change value 2 into 0 in 'related' label.

    Parameters:
    df (pandas.Dataframe): A dataframe to be cleaned.

    Returns:
    pandas.Dataframe:  Cleaned dataframe.
    """

    # Split the categories into separate columns
    categories = df['categories'].str.split(";",expand=True)
    row = categories.iloc[1,]
    category_colnames = row.apply(lambda st : st[0:-2])
    categories.columns = category_colnames

    # Convert the Category values to justo 0 or 1

    for column in categories:
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])

    # Replace categories column in df with new category columns.
    df.drop('categories', axis=1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    # Remove duplciates
    df.drop_duplicates(keep='first', inplace = True)

    # Change value 2 by 0 in the related, because in this problem won't have a difference between those values
    df['related'] = df['related'].apply(lambda x : x%2)

    return df



def save_data(df, database_filename):
    """
    Saves the Dataframe on the received path.

    Parameters:
    df (pandas.DataFrame): Dataframe to be saved.
    database_filename (string): Path to where the Dataframe will be saved.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename[5:-3], engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)


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

    executionTime = time.time() - startTime
    print(f'Execution time in seconds : {executionTime:.3f}')


if __name__ == '__main__':
    main()
