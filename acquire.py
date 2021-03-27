#Wrangle Zillow
#Imports for functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from env import host, user, password 
import os
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

####################ACQUIRE############################

#Connection function to access Codeup mySQL Database and retrieve zillow dataset 
def get_connection(db, user=user, host=host, password=password):
    '''
    This function creates a connection to Codeup Database with 
    info from personal env file (env file has user login information).
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'  

#Function to read sql query and return database

def acquire_zillow():
    '''
    This function reads in the zillow data from the Codeup 
    Database connection made from get_connection
    and returns a pandas DataFrame with all columns.
    '''
    sql_query = '''
                SELECT *
                FROM properties_2017
                JOIN predictions_2017 using(parcelid)
                LEFT JOIN airconditioningtype using (airconditioningtypeid)
                LEFT JOIN architecturalstyletype using (architecturalstyletypeid)
                LEFT JOIN buildingclasstype using (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype using (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype using (propertylandusetypeid)
                LEFT JOIN storytype using (storytypeid)
                LEFT JOIN typeconstructiontype using (typeconstructiontypeid)
                WHERE latitude IS NOT NULL
                AND longitude IS NOT NULL;
                '''

    
    return pd.read_sql(sql_query, get_connection('zillow'))    


def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and 
    writes data to a csv file if cached == False. If cached == True 
    reads in zillow df from a csv file, returns df.
    '''
    if cached == False or os.path.isfile('zillow.csv') == False:
        
        # Read fresh data from db into a DataFrame.
        df = acquire_zillow()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    else:
        
        # If csv file exists or cached == True, read in data from csv.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    return df
