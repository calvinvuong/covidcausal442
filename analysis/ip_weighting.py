import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Logistic regression options
PENALTY_NONE = 'none'
SOLVER_LBFGS = 'lbfgs'
TOL = 1e-4
MAX_ITER = 100


def compute_effects_ip_weighting(
    data: pd.DataFrame,
    stabilize_ip_weights=True,
    logreg_penalty=PENALTY_NONE,
    logreg_solver=SOLVER_LBFGS,
    logreg_tol=TOL,
    logreg_max_iter=MAX_ITER
):
    '''
    Returns a list of results for E[Y | A = a] for increasing values of a using
    parametric IP Weighting.

    Assumes the data is a DataFrame with columns in the order Y, A, L, where:
    - Y is a single continuous outcome variable
    - A is a single dichotomous treatment variable
    - L is a set of continuous treatment variables
    '''
    Y: pd.Series = data.iloc[:, 0]
    A: pd.Series = data.iloc[:, 1]
    L = data.drop(columns=[Y.name, A.name])

    # Predicted class probabilities for each sample
    # shape: (n_samples, n_classes)
    proba = LogisticRegression(
        penalty=logreg_penalty,
        solver=logreg_solver,
        tol=logreg_tol,
        max_iter=logreg_max_iter
    ).fit(L, A).predict_proba(L)

    # 1 / Pr[A = a | L]
    ip_weights = 1 / np.squeeze(
        np.take_along_axis(proba, np.expand_dims(A, axis=1), axis=1)
    )

    if stabilize_ip_weights:
        # Multiply IP Weights by Pr[A = a]
        class_probs = A.value_counts(normalize=True)
        ip_weights *= np.apply_along_axis(
            lambda a: class_probs[a], axis=0, arr=A
        )

    # Calculate weighted E[Y | A = a] for each A = a
    # Store in results by increasing value of a
    results = []
    for treatment in np.sort(A.unique()):
        treatment_indices = np.where(A == treatment)[0]
        outcomes = np.take(Y, treatment_indices)
        weights = np.take(ip_weights, treatment_indices)
        results.append(np.average(outcomes, weights=weights))

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
