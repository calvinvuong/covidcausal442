import argparse
from functools import partial
from utils.analysis.bootstrapping import DIFFERENCE, RATIO
from utils import *

# OUTCOME_VAR = 'Sick'
#OUTCOME_VAR = 'Spread'
OUTCOME_VAR = 'NewCases14Pop'
#TREATMENT_VAR = 'DistancingGrade'
TREATMENT_VAR = 'retail_and_recreation_percent_change_from_baseline'

COVARIATES = [
    'SVISocioeconomic',
    'SVIHousing',
    'StatePctTested',
    'PctGE65',
    'DaytimePopDensity',
    'workplaces_percent_change_from_baseline',
    'Obesity',
    'Smoking',
    'Diabetes'
]


TREATMENT_CUTOFF = 5.0

# Default options
POLYNOMIAL_DEGREE = 2

# Default logistic regression options
PENALTY_NONE = 'none'
SOLVER_LBFGS = 'lbfgs'
TOL = 1e-4
MAX_ITER = 100

# Effect function for using IP weighting. Change args as necessary.
# Default args shown.
effect_func_ipw = partial(
    compute_effects_ip_weighting,
    stabilize_ip_weights=True,
    polynomial_degree=POLYNOMIAL_DEGREE,
    logreg_penalty=PENALTY_NONE,
    logreg_solver=SOLVER_LBFGS,
    logreg_tol=TOL,
    logreg_max_iter=MAX_ITER
)

# Effect function for using standardization. Change args as necessary.
# Default args shown.
effect_func_std = partial(
    compute_effects_standardization,
    polynomial_degree=POLYNOMIAL_DEGREE
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['ipw', 'std'])
    parser.add_argument('samples', type=int)
    args = parser.parse_args()

    effect_func = effect_func_ipw if args.method == 'ipw' else effect_func_std

    # Put data into a Pandas DF
    data = generate_DF('data/', 'population_data.csv', OUTCOME_VAR, TREATMENT_VAR, COVARIATES)

    # Assign DistancingGrade < 5.0 to positive treatment (1),
    # DistancingGrade == 5.0 to no treatment (0)
    #treatment_cutoff = TREATMENT_CUTOFF
    treatment_cutoff = data[TREATMENT_VAR].median()
    dichotomize_treatment(data, TREATMENT_VAR, treatment_cutoff)

    effect_results = bootstrap(
        data, effect_func, n_samples=args.samples, standardize=True
    )

    diff = compute_effect_measure(effect_results, effect_measure=DIFFERENCE)
    ratio = compute_effect_measure(effect_results, effect_measure=RATIO)

    print('\nMethod:', args.method)
    print('Bootstrap Samples:', args.samples)
    print('\nDifference:', diff)
    print('\nRatio:', ratio)
