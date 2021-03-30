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
       FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
       WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL;
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

def count_values(df):
    '''
    This function reads in a dataframe and outputs the column name along 
    with the value counts for each column
    '''
    for column in df.columns:
        if column != 'parcelid'and column != 'id':
            print('Column:', column)
            print(df[column].value_counts())
            print('\n')

def null_city(df):
    '''
    This function takes in a dataframe and outputs all the 
    nulls for each column
    '''
    for column in df.columns:
        print('Column:', column)
        print('Null count:', df[column].isnull().sum())
        print('\n')        

def zillow_dist():
    '''
    This function takes in a dataframe and outputs histograms
    of bedrooms, finished area, logerror, and taxvaluedollarcount
    '''
    
    plt.figure(figsize = (12,8))
    plt.subplot(221)
    plt.hist(df.bedroomcnt)
    plt.title('Bedrooms')



    plt.subplot(222)
    plt.hist(df.calculatedfinishedsquarefeet)
    plt.title('finished area')



    plt.subplot(223)
    plt.hist(df.logerror)
    plt.title('logerror')



    plt.subplot(224)
    plt.hist(df.taxvaluedollarcnt)
    plt.title('taxvaluedollarcnt')

    plt.tight_layout()

def nulls_by_col(df):
    '''
    function that takes in a dataframe of observations and attributes 
    and returns a dataframe where each row is an atttribute name, the 
    first column is the number of rows with missing values for that attribute, 
    and the second column is percent of total rows that have missing values for that attribute
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows': num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing


def cols_missing(df):
    '''
    A function that takes in a dataframe and returns a dataframe with 3 columns: 
    the number of columns missing, 
    percent of columns missing, 
    and number of rows with n columns missing
    '''
    df2 = pd.DataFrame(df.isnull().sum(axis =1), columns = ['num_cols_missing']).reset_index()\
    .groupby('num_cols_missing').count().reset_index().\
    rename(columns = {'index': 'num_rows' })
    df2['pct_cols_missing'] = df2.num_cols_missing/df.shape[1]
    return df2

def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
    '''
    A function that will drop rows or columns based on the percent of values that are missing: 
    handle_missing_values(df, prop_required_column, prop_required_row)
    '''
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

def remove_columns(df, cols_to_remove):  
    '''
    A function that will drop columns you want removed from dataframe
    '''
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    '''
    A function that will handle erroneous data, handle missing values,
    remove columns, add columns, replace nulls, fill nulls, and drop nulls
    for zillow dataset
    '''
    df = pd.read_csv('zillow.csv')
    
    # Restrict df to only properties that meet single unit use criteria
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    
    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles',
                           np.where(df.fips == 6059, 'Orange', 
                                   'Ventura'))    
    # drop columns not needed
    df = remove_columns(df, ['id',
       'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'heatingorsystemtypeid'
       ,'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 
        'censustractandblock', 'propertylandusedesc'])


    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)
    
    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)
    
    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    df = df[df.taxvaluedollarcnt < 5_000_000]
    df[df.calculatedfinishedsquarefeet < 8000]
    
    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    
    return df
