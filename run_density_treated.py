from utils import *
import sys
from functools import partial

OUTCOME_VAR = 'Sick'
TREATMENT_VAR = 'DaytimePopDensity'
COVARIATES = ['SVISocioeconomic', 'StatePctTested', 'PctGE65', 'DistancingGrade']
TREATMENT_CUTOFF = 300.0 # CHANGED TO MEAN OF WHOLE DATASET, Need to explore further

# Default options
POLYNOMIAL_DEGREE = 2

# Default logistic regression options
PENALTY_NONE = 'none'
PENALTY_NONE = 'none'
SOLVER_LBFGS = 'lbfgs'
TOL = 1e-4
MAX_ITER = 100

# Effect function for using IP weighting. Change args as necessary. Default args shown.
effect_function_ip = partial(compute_effects_ip_weighting,
                             stabilize_ip_weights=True,
                             polynomial_degree=POLYNOMIAL_DEGREE,
                             logreg_penalty=PENALTY_NONE,
                             logreg_solver=SOLVER_LBFGS,
                             logreg_tol=TOL,
                             logreg_max_iter=MAX_ITER)

# Effect function for using standardization. Change args as necessary. Default args shown.
effect_function_stand = partial(compute_effects_standardization,
                                polynomial_degree=POLYNOMIAL_DEGREE)


def run_analysis(func=effect_function_ip, effect_measure='difference', num_samples=100):
    # Put data into a Pandas DF
    data = generate_DF('data/', OUTCOME_VAR, TREATMENT_VAR, COVARIATES)
    # Assign DistancingGrade < 5.0 to positive treatment (1), DistancingGrade == 5.0 to no treatment (0)
    dichotomize_treatment(data, TREATMENT_VAR, TREATMENT_CUTOFF)
        
    # Compute causal expectations
    expectations = bootstrap(data, func, n_samples=num_samples)
    effect = compute_effect_measure(expectations, effect_measure)
    return effect



if __name__ == '__main__':
    # Parse command line inputs.
    # For now, only take method, measure,  and number of bootstrap samples as args
    if len(sys.argv) != 4:
        print("Usage: python3 run.py <method> <measure> <bootstrap samples>")
        print("method: ip or standardization")
        print("measure: difference or ratio")
        print("bootstrap samples: number of bootstrap samples to run")
        sys.exit(0)
    else:
        effect_func = effect_function_ip if sys.argv[1] == 'ip' else effect_function_stand
        causal_effect = run_analysis(func=effect_func, effect_measure=sys.argv[2], num_samples=int(sys.argv[3]))
        print("Method: ", sys.argv[1])
        print("Measure: ", sys.argv[2])
        print("Bootstrap samples: ", int(sys.argv[3]))
        print(causal_effect)
