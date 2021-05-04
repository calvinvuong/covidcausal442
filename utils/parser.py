import pandas as pd
import numpy as np
import os
from typing import List


def generate_DF(
    data_dir: str,
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
    # Columns to include in DataFrame, with 'name' being temporary
    select_columns = ['name', outcome_name, treatment_name] + covariate_names

    # Read each data file into a separate DataFrame
    df_list = []
    for data_file in os.listdir(data_dir):
        data_file_path = os.path.join(data_dir, data_file)
        # Read data into Pandas DataFrame
        df = pd.read_csv(data_file_path, header=0)
        # Rename first column to 'date'
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)

        df_list.append(df[select_columns])

    # Merge all DFs into one
    combined_df = pd.concat(df_list)
    # Drop any rows with missing data
    combined_df.dropna(inplace=True)
    # Select only rows for Ohio counties
    ohio_df = combined_df[combined_df['name'].str.contains('Ohio,')]
    # Drop the county names
    ohio_df = ohio_df.drop(['name'], axis=1)
    ohio_df.reset_index(drop=True, inplace=True)

    ohio_df['Spread'] = 1 / 10 ** ohio_df['Spread']

    return ohio_df


def dichotomize_treatment(
    data: pd.DataFrame,
    treatment_name: str,
    thresh: float
):
    '''
    Takes as input a Pandas DF, a column name 'treatment_name' to dichotomize,
    and a threshold. Values >= thresh are assigned to 0; values < thresh are
    assigned to 1.
    '''
    data[treatment_name] = np.where(data[treatment_name] >= thresh, 0, 1)


def test():
    '''Function for testing.'''
    data_dir = '../data/'
    outcome_name = 'Sick'
    treatment_name = 'DistancingGrade'
    covariate_names = [
        'SVISocioeconomic', 'StatePctTested', 'PctGE65', 'DaytimePopDensity'
    ]
    df = generate_DF(data_dir, outcome_name, treatment_name, covariate_names)
    print(df)
    dichotomize_treatment(df, 'DistancingGrade', 5.0)
    print(df)


if __name__ == '__main__':
    test()
