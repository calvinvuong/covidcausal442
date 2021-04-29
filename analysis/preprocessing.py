import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


POLYNOMIAL_DEGREE = 2


def standardize_data(data: pd.DataFrame):
    '''
    Rescales the covariates L in the data to have a mean of 0 and a standard
    deviation of 1. Performs the operation in place.

    Assumes the data is a DataFrame with columns in the order Y, A, L, where:
    - Y is a single continuous outcome variable
    - A is a single dichotomous treatment variable
    - L is a set of continuous treatment variables
    '''
    Y: pd.Series = data.iloc[:, 0]
    A: pd.Series = data.iloc[:, 1]
    L = data.drop(columns=[Y.name, A.name])
    data[L.columns] = StandardScaler().fit_transform(L)


def polynomialize_data(
    data: pd.DataFrame,
    degree=POLYNOMIAL_DEGREE,
    include_treatment=False
):
    '''
    Generates polynomial and interaction features of a given degree. Includes
    only the covariates L unless include_treatment is specified. Performs the
    operation in place.

    Assumes the data is a DataFrame with columns in the order Y, A, L, where:
    - Y is a single continuous outcome variable
    - A is a single dichotomous treatment variable
    - L is a set of continuous treatment variables
    '''
    Y: pd.Series = data.iloc[:, 0]

    # Select the columns to exclude from polynomialization
    drop_columns = [Y.name]
    if not include_treatment:
        A: pd.Series = data.iloc[:, 1]
        drop_columns.append(A.name)

    # Covariate data that will be polynomialized
    L = data.drop(columns=drop_columns)

    # Drop the covariate columns from the original data
    data.drop(columns=L.columns, inplace=True)

    # Polynomialize the covariates and add these columns back to the data
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    data[poly.get_feature_names(L.columns)] = poly.fit_transform(L)


def _construct_test_dataset():
    '''Creates a test dataset.'''
    data = {
        'Y': [14.01, 90.20, 90.95, 96.27, 91.21],
        'A': [0, 0, 1, 0, 1],
        'L1': [103.02, 107.26, 110.35, 114.23, 114.68],
        'L2': [32.23, 39.44, 45.48, 32.61, 34.69]
    }
    return pd.DataFrame(data=data)
