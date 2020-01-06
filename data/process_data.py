import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages, categories):
    """
    Load & Merge Messages and Categories 
    --
    Inputs:
        messages:  messages data in csv file
        categories: categories data in csv file 
    Outputs:
        df: Dataframe Merged with Messages and Categories 
    """
    # load messages
    messages_dataset = pd.read_csv(messages)
    # load categories
    categories_dataset = pd.read_csv(categories)
    # merge messages & categories
    merged_dataset = messages_dataset.merge(categories_dataset, on = 'id')
    
    return merged_dataset

def clean_data(df):
    """
    Clean data by applying spliting, droping, removing duplicates.
    --
    Inputs:
        df: dataset of Messages and Categories 
    Outputs:
        df: Cleaned Dataframe
    """
    categories = df.categories.str.split(pat=';', expand=True)
    # extract a list of new column names for categories
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    save data into sqlLite Database
    --
    Inputs:
        df: dataset of Messages and Categories 
        database_filename: database name

    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('MessagesDataSet', engine, index=False, if_exists='replace')  


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


if __name__ == '__main__':
    main()