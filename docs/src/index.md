# FKRBDemand.jl

[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://jamesbrand.github.io/FKRBDemand.jl/)

This package implements a simple version of the method introduced by [Fox, Kim, Ryan, and Bajari (2009)](https://onlinelibrary.wiley.com/doi/abs/10.3982/QE49) (FKRB) for estimating random coefficient mixed logit demand models with market-level data. This approach allows us to estimate the distribution of consumer preferences nonparametrically and through a simple elastic net regression, thereby avoiding some of the convergence and speed issues with empirical estimation of random coefficient models. The package allows you to do the following with just a few lines of code: 

- Estimate a random coefficient logit model 
- Run a bootstrap to get standard errors on the resulting model estimates (i.e., estimated weights) 
- Calculate price elasticities at existing prices (easy todo: allow for counterfactual prices)
- Plot the (nonparametric) distribution of random coefficients

Currently, I've only implemented the FKRB approach for market-level data. My goal is to have an API which is familiar and relatively consistent across [FRACDemand.jl](github.com/jamesbrandecon/FRACDemand.jl), [NPDemand.jl](github.com/jamesbrandecon/NPDemand.jl), and any other code I share for demand estimation, so that multiple packages can be tested quickly and eventually the packages can be merged together. PRs welcome -- without them, updates will be very slow.  

## Installation 
```julia
using Pkg; 
# FRAC has to be installed first
Pkg.add("FRACDemand") 
# Install FKRBDemand from Github: 
Pkg.add(url = "https://github.com/jamesbrandecon/FKRBDemand.jl")
```

## Usage 
```julia
using FKRBDemand

df = ... # DataFrame with columns:
# "x" (any number of product characteristics),
# "prices",
# "shares" (market shares),
# "demand_instruments0" (any number of demand instruments)
# "market_ids" 
# "product_ids"
```

Estimation is then straightforward: 
```julia
problem = FKRBDemand.define_problem( 
        data = df, 
        linear = ["prices", "x"], # in FKRB, linear and nonlinear should be the same
        nonlinear = ["prices", "x"], 
        train = [], # the set of market_ids to use for tuning regularization, mostly not yet useful
        fixed_effects = ["product_ids"], # strings of column names in df to aborb in FRAC
        alpha = 0.01, # FKRB grid will aim to cover (1-alpha*100)% of the distribution, based on FRACDemand estimates
        step = 0.1 # step size for the FKRB grid
        );

FKRBDemand.estimate!(problem, 
    method = "elasticnet", # only useful method implemented so far
    constraints = [:nonnegativity, :proper_weights], # constraints on the weights
    lambda = 1e-6) # regularization strength
```

## Price endogeneity
The `FKRB` approach is best justified when all product characteristics are exogenous, but we are often interested in settings where that is not the case. The most general approach I've seen to handle endogeneity within this estimation approach is that from Meeker (2021), so I implemented a slightly modified version of his approach. I first estimate the model using `FRACDemand.jl`, using the same problem specifications that have been provided to `FKRBDemand.jl`, meaning that the regression FRACDemand.jl uses allows for random coefficients and instrument for prices. Then, I store the estimated demand shocks from that procedure and include them in the market-level demand function when estimating through `FKRBDemand.jl`. The intuition for this approach is that, by generating a good first estimate of unobserved demand shocks, we can then control for those shocks in the second stage. If we think of the regression errors in the FKRB second stage as measurement error, then this approach avoids the omitted variable bias (endogeneity) induced by running `FKRBDemand.jl` naively without correcting for endogenous prices. 

If you have an alternative preferred approach to estimating demand shocks, simply include a "xi" field in the data you provide to `define_problem` and these will be included in the utility function automatically. If you need fixed effects too, estimate them first via FRAC or a logit specification, add them together with your residuals `xi`, and include the sum as the field "xi" in `problem.data`.

## Inference
I wasn't sure how to do inference here, but [Meeker (2021)](https://www.imeeker.com/files/jmp.pdf) recommends subsampling/bootstrapping. I've implemented two simple helper functions for this purpose: `bootstrap!` and `subsample!`, which implement the inference approaches corresponding to the function names. Both functions take the `FKRBProblem` as the sole required argument, plus keyword arguments controlling how many times to repeat the estimation procedure and how large a subsample to use. After one of these functions are called, `problem.std` will contain the implied standard errors from the called approach. How to make the resulting standard errors easy to use is a work in progress, but `problem.results["boot_weights"]` will contain the bootstrapped weights, which can be used to calculate confidence intervals on objects of interest.

## Visualization
The best function for visualization is now `plot_coefficients`, which plots a grid of distributions. The diagonal of this grid shows kernel density estimates of the corresponding coefficients, while the off-diagonal elements show the estimated covariance between the coefficients. The function `plot_coefficients` takes a `FKRBProblem` object as its only required argument, though the heatmaps, densities, and the set of coefficients to plot can all be controlled through keyword arguments as shown below.

```julia
    plot_coefficients(
        problem;
        select_dims = ["prices", "x"],
        heatmap_kwargs   = (c = cgrad([:white, :lightblue, :blue]), alpha = 0.6),
        marg_kwargs      = (c = :lightcoral, lw=2, alpha = 0.8, nbins=50)
    )
```