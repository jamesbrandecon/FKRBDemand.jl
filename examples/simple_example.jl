using FRACDemand, FKRBDemand
using Distributions, Plots
using FixedEffectModels, DataFrames
using IterTools: product

preference_means = [-1.0 1.0]; # means of the normal distributions of preferences -- 
# first column is preference for price, second is x
preference_SDs = [sqrt(0.3) sqrt(0.3)]; # standard deviations of the normal distributions of preferences

J1, J2, T, B = 20, 20, 300, 1
β = preference_means
variances = preference_SDs.^2
ρ = 0.9
covariance = ρ * preference_SDs[1] * preference_SDs[2]
Σ = [variances[1] covariance; covariance variances[2]]
ξ_var = 0.3

df = FRACDemand.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
df[df.market_ids .>T,:product_ids] .+= J1 - 2
df_original = copy(df)
df = select(df, Not(:xi))
df[!,"demand_instruments3"] = df[!,"demand_instruments1"] .* df[!,"demand_instruments2"]
# If you don't provide domain ranges, define_problem will run FRAC.jl with the same problem specs
# and will use the estimated variance of preferences to define a grid width 
# that should cover 99.9%, i.e. (1-alpha)%, of the preference domain based on those estimates
problem = FKRBDemand.define_problem( 
        data = df, 
        linear = ["prices", "x"], 
        nonlinear = ["prices", "x"], 
        fixed_effects = ["product_ids"],
        alpha = 0.001, 
        step = 0.1
        );

# If you want to define the grid width yourself, 
# you can do so by providing a range
# Doing this avoids the inference issues with preestimating the grid
problem = FKRBDemand.define_problem( 
            data = df, 
            linear = ["prices", "x"],
            nonlinear = ["prices", "x"],
            fixed_effects = ["dummy_FE"],
            range = Dict("x" => 0:0.1:4, "prices" => -4:0.1:0));

# One-line estimation
FKRBDemand.estimate!(problem, 
    method = "elasticnet",
    constraints = [:nonnegativity, :proper_weights],
    gamma = 0.0, # L1 penalty
    lambda = 0.0) # L2 penalty

FKRBDemand.estimate!(problem, 
    method = "elasticnet",
    constraints = nothing,
    silent = true,
    lambda = range(1e-6, 0.1, 2), 
    cross_validate = true) 

# -------------------------------
# Inference + post-estimation
# -------------------------------
FKRBDemand.subsample!(problem; n_samples = 500)
FKRBDemand.bootstrap!(problem, n_samples = 5)

# You can then pull the estimated weights and standard errors for individual regression coefficients: 
parameters = problem.results["weights"]; # parameters will contain the weights associated with each grid point
std = problem.std; # Only valid after running subsample! or boostrap!-- 

# Can calculate all price elasticities into a DataFrame  
FKRBDemand.price_elasticities!(problem)
elasticities_df = FKRBDemand.elasticities_df(problem)


true_own_elasticities = zeros(size(df,1))
truth = FRACDemand.sim_true_price_elasticities(df_original, dropdims(β, dims=1), Σ)
truth[truth.product_i .== truth.product_j,:]
own_elasticities = elasticities_df[(elasticities_df.product1 .== elasticities_df.product2),:]


# ----------------
# Plots
# ----------------
# CDF
cdf_plot = plot_cdfs(problem)
# add true CDFs
plot!(cdf_plot, unique(problem.grid_points[:,1]), cdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1])), color = :blue, ls = :dash, label = "CDF 1, true")
plot!(cdf_plot, unique(problem.grid_points[:,2]), cdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2])), color = :red, ls = :dash, label = "CDF 2, true")

# PDF
pmf_plot = plot_pmfs(problem)
# add true PMFs 
plot!(pmf_plot, unique(problem.grid_points[:,1]), pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1])) ./sum(pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1]))), color = :blue, ls = :dash, label = "PDF 1, true")
plot!(pmf_plot, unique(problem.grid_points[:,2]), pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2])) ./sum(pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2]))), color = :red, ls = :dash, label = "PDF 2, true")

# Histogram of estimated and true elasticities
histogram(
    own_elasticities.elast, 
    label = "Estimated", 
    xlabel = "Own-Price Elasticity", 
    ylabel = "Count", 
    color = :skyblue,
    alpha = 0.5, 
    normalize = :density
    )

histogram!(
    truth[truth.product_i .== truth.product_j,:elasticity], 
    label = "Truth", 
    color = :lightcoral, 
    alpha = 0.5, 
    normalize = :density
    )

# Scatterplot of estimated vs true elasticities
scatter(
    own_elasticities.elast, 
    truth[truth.product_i .== truth.product_j,:elasticity], 
    label = "", 
    color = :lightcoral, 
    alpha = 0.5,
    xlabel = "Estimated Own-Price Elasticity",
    ylabel = "True Own-Price Elasticity"
)
plot!(own_elasticities.elast, own_elasticities.elast, 
    label = "45 degree line", color = :black, ls = :dash, 
    lw = 3)

# Joint heatmap of estimated coefficients
plot_coefficients(
        problem;
        select_dims = ["prices", "x"],
        heatmap_kwargs   = (c = cgrad([:white, :lightblue, :blue]), alpha = 0.6),
        marg_kwargs      = (c = :lightcoral, lw=2, alpha = 0.8, nbins=50)
    )