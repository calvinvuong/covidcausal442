import pandas as pd
import numpy as np
import os
from typing import List
from datetime import timedelta
import sys

# Read population csv file and return a Pandas DF
def read_population(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, header=0)
    return df
    
def generate_DF(
    data_dir: str,
    pop_file: str,
    outcome_name: str,
    treatment_name: str,
    covariate_names: List[str]
) -> pd.DataFrame:
    '''
    Reads data files into Pandas DataFrame.
    Column 1: Outcome Y (number of cases)
    Column 2: Treatment A (mobility score)
    Other columns: Covariates L
    '''
    # Columns to include in DataFrame, with 'date', 'name', 'casrn' being temporary
    if outcome_name == 'NewCases14Pop': 
        select_columns = ['date', 'name', 'casrn', 'Cases', treatment_name] + covariate_names
    elif treatment_name == 'NewCases14Pop':
        select_columns = ['date', 'name', 'casrn', 'Cases', outcome_name] + covariate_names
    else:
        select_columns = ['date', 'name', 'casrn', 'Cases', outcome_name, treatment_name] + covariate_names

    # Read each data file into a separate DataFrame
    df_list = []
    for data_file in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, data_file)
        # Read data into Pandas DataFrame
        df = pd.read_csv(data_file_path, header=0)
        # Rename first column to 'date' and format to datetime
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

        df_list.append(df[select_columns])

    # Concatenate all DFs into one
    combined_df = pd.concat(df_list)
    # Drop any rows with missing data
    combined_df.dropna(inplace=True)
    # Select only rows for Ohio counties
    ohio_df = combined_df[combined_df['name'].str.contains('Ohio, ')]
    
    # Read population data and join into same DataFrame
    pop_df = pd.read_csv(pop_file, header=0)
    ohio_df = ohio_df.merge(pop_df)

    # Do extra work to compute number of new cases in 14 days since treatment,
    # population adjusted
    if outcome_name == 'NewCases14Pop' or treatment_name == 'NewCases14Pop':
        # Create a temporary dataframe with case numbers 14 days after treatment
        tmp_df = ohio_df[['date', 'casrn', 'Cases']]
        if outcome_name == 'NewCases14Pop':
            tmp_df.loc[:,'date'] = tmp_df['date'] + pd.DateOffset(days=-14) # Shift date by 14 days
        else: # is treatment
            tmp_df.loc[:,'date'] = tmp_df['date'] + pd.DateOffset(days=14) # Shift date by 14 days
        # Join dfs to "match" 14-day shifted dates
        ohio_df = ohio_df.merge(tmp_df[['date', 'casrn', 'Cases']], on=['date', 'casrn'])
        # Compute new cases
        ohio_df['NewCases14Pop'] = abs(ohio_df['Cases_y'] - ohio_df['Cases_x']) / ohio_df['population'] * 100000

    # Select only desired columns
    return ohio_df[[outcome_name, treatment_name] + covariate_names]


def dichotomize_treatment(
    data: pd.DataFrame,
    treatment_name: str,
    thresh: float,
    gt_treatment: int=0    
):
    '''
    Takes as input a Pandas DF, a column name 'treatment_name' to dichotomize,
    , a threshold, and gt_treatment. gt_treatment should either be 0 or 1.
    Values >= thresh are assigned to gt_treatment; values < thresh are
    assigned to 1-gt_treatment. By default, gt_treatment = 0
    '''
    assert (gt_treatment == 0 or gt_treatment == 1)
    data[treatment_name] = np.where(data[treatment_name] >= thresh, gt_treatment, 1-gt_treatment)
    

def test():
    '''Function for testing.'''
    data_dir = '../data/'
    pop_file = '../population_data.csv'
    outcome_name = 'NewCases14Pop'
    treatment_name = 'retail_and_recreation_percent_change_from_baseline'
    covariate_names = [
        'SVISocioeconomic', 'StatePctTested', 'PctGE65', 'DaytimePopDensity'
    ]
    df = generate_DF(data_dir, pop_file, outcome_name, treatment_name, covariate_names)
    dichotomize_treatment(df, treatment_name, df[treatment_name].median())
    print(df)

if __name__ == '__main__':
    test()
