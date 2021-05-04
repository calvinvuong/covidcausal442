import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .preprocessing import polynomialize_data


# Degree of polynomial and interaction features
POLYNOMIAL_DEGREE = 2


def compute_effects_standardization(
    data: pd.DataFrame,
    polynomial_degree=POLYNOMIAL_DEGREE
):
    '''
    Returns a list of results for E[Y | A = a] for increasing values of a using
    parametric Standardization.

    Assumes the data is a DataFrame with columns in the order Y, A, L, where:
    - Y is a single continuous outcome variable
    - A is a single dichotomous treatment variable
    - L is a set of continuous treatment variables
    '''
    Y: pd.Series = data.iloc[:, 0]
    A: pd.Series = data.iloc[:, 1]

    # Polynomialize the covariates and treatment
    poly_data = data.copy()
    polynomialize_data(poly_data, degree=polynomial_degree, include_A=True)

    # Inputs to regression (treatment columns A and covariate columns L)
    AL = poly_data.drop(columns=[Y.name])
    # Fit regression to Y
    linreg = LinearRegression().fit(AL, Y)

    # Calculate weighted E[Y | A = a] for each A = a
    # Store in results by increasing value of a
    results = []
    for a in np.sort(A.unique()):
        # Create new data with uniform treatment value A = a
        data_a = data.copy()
        data_a[A.name] = np.full(A.shape, a)
        # Polynomialize data
        polynomialize_data(data_a, degree=polynomial_degree, include_A=True)

        # Get only treatment and covariate columns for prediction
        AL_a = data_a.drop(columns=[Y.name])
        # Mean outcome in new data is standardized mean outcome
        results.append(np.mean(linreg.predict(AL_a)))

    return results
