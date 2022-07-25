import arviz as az
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pragmatics import RSA
from pragmatics import nonRSA

def optimize(y_empirical, model_to_optimize):

    # Initialize random number generator
    RANDOM_SEED = 8927
    rng = np.random.default_rng(RANDOM_SEED)
    az.style.use("arviz-darkgrid")

    import pymc as pm
    basic_model = pm.Model()

    with basic_model:

        # Priors for unknown model parameters: we have beta as our parameter
        beta = pm.Normal("beta", mu=0, sigma=10, shape=2)

        # Expected value of outcome
        if model_to_optimize == "nonRSA":
            # compute a "mu" for the nonRSA model
            mu = nonRSA.speaker_targetboard(boards[boardname], alpha, beta, cluelist_union, representations, 'swow', vocab, target_df)
            # Likelihood (sampling distribution) of observations
            Y_obs = pm.Normal("Y_obs", mu=mu, observed=y_empirical)

    with basic_model:
        # draw 1000 posterior samples
        idata = pm.sample()

    return az.summary(idata, round_to=2)