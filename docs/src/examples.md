# Examples

First, we can simulate some data using the `FRACDemand` package. The following example simulates a dataset with 600 markets, each with either 10 or 20 products. The resulting data frame contains columns for product characteristics, prices, shares, demand instruments, market IDs, and product IDs. 

```julia 
    # Half of markets will have 10 products, half will have 20 products
    # There will be 600 (300*2) markets in total
    J1, J2, T = 10, 20, 300  

    β = [-1.0 1.0] # mean random coefficients
    variances = [0.3.^2 0.3^2] # variances of random coefficients
    ρ = 0.9
    covariance = ρ * preference_SDs[1] * preference_SDs[2]
    Σ = [variances[1] covariance; covariance variances[2]]
    
    ξ_var = 0.3 # 

    # Use FRACDemand function to generate the data
    df = FRACDemand.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
    df[df.market_ids .>T,:product_ids] .+= J1 - 2 # this is correcting a bug -- necessary but will be fixed soon
    
    df = select(df, Not(:xi)) # remove simulated xi -- will be replaced in estimation

    # Simulated data only has two demand instruments, but we are going to add 
    df[!,"demand_instruments3"] = df[!,"demand_instruments1"] .* df[!,"demand_instruments2"]
```

Then we "define" a problem using the `FKRBDemand` package. The `define_problem` function takes a data frame and a set of arguments that specify the problem to be solved. The arguments include the data frame, the linear and nonlinear variables, the training set, fixed effects, alpha, and step size. The function returns a `FKRBProblem` object that contains the problem definition.

```julia
    # Define the problem
    problem = FKRBDemand.define_problem( 
        data = df, 
        linear = ["prices", "x"], # in FKRB, linear and nonlinear should be the same
        nonlinear = ["prices", "x"], 
        train = [], # the set of market_ids to use for tuning regularization, mostly not yet useful
        fixed_effects = ["product_ids"], # strings of column names in df to aborb in FRAC
        alpha = 0.01, # FKRB grid will aim to cover (1-alpha*100)% of the distribution, based on FRACDemand estimates
        step = 0.1 # step size for the FKRB grid
    )
```

For estimation, the only required input is the problem object, though you can also manually control the regularization strength and how to constrain the weights. 

```julia
    # Estimate the model
    FKRBDemand.estimate!(problem, 
        method = "elasticnet", # only useful method implemented so far
        constraints = [:nonnegativity, :proper_weights], # constraints on the weights
        lambda = 1e-6) # regularization strength
```

There are a couple of plotting tools we can use. If you want to plot the joint distribution of random coefficients, you can use the `plot_coefficients` function.

```julia
    # Plot the coefficients
    plot_coefficients(
        problem;
        select_dims = ["prices", "x"],
        heatmap_kwargs   = (c = cgrad([:white, :lightblue, :blue]), alpha = 0.6),
        marg_kwargs      = (c = :lightcoral, lw=2, alpha = 0.8, nbins=50)
    )
```

We also store the CDF in the `results` field of the `FKRBProblem` object. This can be used to plot the CDF of the random coefficients, and easy transformations of this can be used to make custom plots of distributions. 

```julia
    # Plot the CDF of the random coefficients
    using Plots
    cdf_dataframe = problem.results["cdf_dataframe"]
    # Dataframe in this case has three columns: "prices", "x", and "values"
    # Values denotes the x-axis of the CDF, and each column denotes the CDF of the corresponding coefficient at that value
    
    # How to plot the CDF of price coefficients
    plot(
        cdf_dataframe.values, 
        cdf_dataframe.prices,
        xlabel = "Random Coefficient on Price",
        ylabel = "Cumulative Probability"
    )
```

Finally, we can use the `subsample!` and `bootstrap!` functions to perform inference on the estimated coefficients. These functions take the `FKRBProblem` object as their only required argument, plus keyword arguments controlling how many times to repeat the estimation procedure and how large a subsample to use. After one of these functions are called, `problem.std` will contain the implied standard errors from the called approach.

```julia
    # Subsampling
    FKRBDemand.subsample!(problem, n_samples=100, sample_size=0.8)
    
    # Bootstrapping
    FKRBDemand.bootstrap!(problem, n_samples=100)
``` 

Currently, the use of the resulting bootstraps/subsamples is limited (e.g., I'm not yet passing the uncertainty through to price elasticities). One thing you can do is add the uncertainty to `plot_coefficients` by passing `include_CI` set to `true`. This will add the marginal distributions of the bootstrapped/subsampled coefficients to the plot. 
