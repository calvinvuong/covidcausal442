import argparse
from functools import partial
from utils.analysis.bootstrapping import DIFFERENCE, RATIO
from utils import *

OUTCOME_VAR = 'NewCases14Pop'
TREATMENT_VAR = 'retail_and_recreation_percent_change_from_baseline'
# Covariates to adjust on
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
    parser.add_argument('treatment', nargs='?', default=TREATMENT_VAR)
    parser.add_argument('outcome', nargs='?', default=OUTCOME_VAR)
    args = parser.parse_args()

    effect_func = effect_func_ipw if args.method == 'ipw' else effect_func_std


    # Put data into a Pandas DF and manipulate
    data = generate_DF('data/', 'population_data.csv', args.outcome, args.treatment, COVARIATES)

    # Change treatment variable into discrete values
    treatment_cutoff = data[args.treatment].median()
    if 'change_from_baseline' in args.treatment:
        # For this treatment, values > cutoff are considered untreated
        dichotomize_treatment(data, args.treatment, treatment_cutoff, gt_treatment=0)
    else:
        # For all other treatments, values > cutoff are considered treated
        dichotomize_treatment(data, args.treatment, treatment_cutoff, gt_treatment=1)

    # Compute causal expectations
    effect_results = bootstrap(
        data, effect_func, n_samples=args.samples, standardize=True
    )

    # Compute effect measures
    diff = compute_effect_measure(effect_results, effect_measure=DIFFERENCE)
    ratio = compute_effect_measure(effect_results, effect_measure=RATIO)

    # Print results
    print("\nTreatment:", args.treatment)
    print("Outcome:", args.outcome)
    print('Method:', args.method)
    print('Bootstrap Samples:', args.samples)
    print('\nDifference:', diff)
    print('Ratio:', ratio)
    print('--------------------------------------------------------------------------------------------')
