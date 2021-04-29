import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_effects_standardization(data: pd.DataFrame):
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

    # Inputs to regression (treatment columns A and covariate columns L)
    AL = data.drop(columns=[Y.name])
    # Fit regression to Y
    linreg = LinearRegression().fit(AL, Y)

    # Calculate weighted E[Y | A = a] for each A = a
    # Store in results by increasing value of a
    results = []
    for a in np.sort(A.unique()):
        # Create new data with uniform treatment value A = a
        AL_a = data.drop(columns=[Y.name, A.name])
        AL_a.insert(0, A.name, np.full(A.shape, a))
        # Mean outcome in new data is standardized mean outcome
        results.append(np.mean(linreg.predict(AL_a)))

    return results


def _construct_test_dataset():
    '''Creates a test dataset based on Table 2.2 in Hernan-Robins text.'''
    data = {
        'Y': [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        'A': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'L1': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        # 'L2': [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
        # 'L3': [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
    }
    index = [
        'Rheia', 'Kronos', 'Demeter', 'Hades', 'Hestia', 'Poseidon', 'Hera',
        'Zeus', 'Artemis', 'Apollo', 'Leto', 'Ares', 'Athena', 'Hephaestus',
        'Aphrodite', 'Cyclope', 'Persephone', 'Hermes', 'Hebe', 'Dionysus'
    ]
    return pd.DataFrame(data=data, index=index)
