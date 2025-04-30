using FRACDemand, FKRBDemand
using Distributions, Plots
using FixedEffectModels, DataFrames
using IterTools: product

preference_means = [-1.0 1.0]; # means of the normal distributions of preferences -- 
# first column is preference for price, second is x
preference_SDs = [0.3 0.3]; # standard deviations of the normal distributions of preferences

J1, J2, T, B = 20, 20, 500, 1
β = preference_means
Σ = [preference_SDs[1] 0.0; 0.0 preference_SDs[2]]
ξ_var = 0.3

df = FRACDemand.sim_logit_vary_J(J1, J2, T, B, β, Σ, ξ_var)
df[df.market_ids .>T,:product_ids] .+= J1 - 2
df_original = copy(df)
df = select(df, Not(:xi))

# If you don't provide domain ranges, define_problem will run FRAC.jl with the same problem specs
# and will use the estimated variance of preferences to define a grid width 
# that should cover 99% of the preference domain based on those estimates
problem = FKRBDemand.define_problem( 
        data = df, 
        linear = ["prices", "x"], 
        nonlinear = ["prices", "x"], 
        train = collect(1:300),
        fixed_effects = ["product_ids"],
        alpha = 0.01, 
        step = 0.1 
        );

# # If you want to define the grid width yourself, you can do so by providing a range
problem = FKRBDemand.define_problem( 
            data = df, 
            linear = ["prices", "x"],
            nonlinear = ["prices", "x"],
            fixed_effects = ["dummy_FE"],
            train = collect(1:300),
            range = Dict("x" => -4:0.1:0, "prices" => -4:0.1:0));

# One-line estimation
FKRBDemand.estimate!(problem, 
    method = "elasticnet",
    gamma = 1e-5, # L1 penalty
    lambda = 1e-6) # L2 penalty

FKRBDemand.estimate!(problem, 
    method = "elasticnet",
    constraints = nothing,
    silent = true,
    lambda = range(1e-6, 0.1, 2), 
    cross_validate = true) 

# ----------------
# Plots
# ----------------
# CDF
cdf_plot = plot_cdfs(problem)
# add true CDFs
plot!(cdf_plot, unique(problem.grid_points[:,1]), cdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1])), color = :blue, ls = :dash, label = "CDF 1, true")
plot!(cdf_plot, unique(problem.grid_points[:,2]), cdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2])), color = :red, ls = :dash, label = "CDF 2, true")


pmf_plot = plot_pmfs(problem)
# add true PMFs 
plot!(pmf_plot, unique(problem.grid_points[:,1]), pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1])) ./sum(pdf.(Normal(preference_means[1],preference_SDs[1]), unique(problem.grid_points[:,1]))), color = :blue, ls = :dash, label = "PDF 1, true")
plot!(pmf_plot, unique(problem.grid_points[:,2]), pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2])) ./sum(pdf.(Normal(preference_means[2],preference_SDs[2]), unique(problem.grid_points[:,2]))), color = :red, ls = :dash, label = "PDF 2, true")

# -------------------------------
# Inference + post-estimation
# -------------------------------
FKRBDemand.subsample!(problem; n_samples = 5)
FKRBDemand.bootstrap!(problem, n_samples = 20)

# You can then pull the estimated weights and standard errors for individual regression coefficients: 
parameters = problem.results[1]; # parameters will contain the weights associated with each grid point
std = problem.std; # Only valid after running subsample! or boostrap!-- 

# std will contain the bootstrapped/subsampled standard error 
grid = rename(DataFrame(problem.grid_points, :auto), :x1 => :x, :x2 => :prices);

# Can calculate all price elasticities into a DataFrame  
FKRBDemand.price_elasticities!(problem)
elasticities_df = FKRBDemand.elasticities_df(problem)

function sim_true_price_elasticities(df::DataFrame, beta::AbstractVector, Σ::AbstractMatrix;
    price_col::Symbol=:prices,
    x_col::Symbol=:x,
    xi_col::Symbol=:xi)
    out = DataFrame(market_ids=Int[], product_i=Int[], product_j=Int[], elasticity=Float64[])
    for subdf in groupby(df, :market_ids)
        p  = subdf[!, price_col]
        x  = subdf[!, x_col]
        xi = subdf[!, xi_col]
        fe = hasproperty(subdf, :market_FEs) ? first(subdf.market_FEs) : 0
        E  = FRACDemand.sim_price_elasticities(p, x, xi, beta, Σ; market_FE=fe)
        mid, products = first(subdf.market_ids), unique(subdf.product_ids)
        for i in eachindex(products), j in eachindex(products)
            push!(out, (market_ids=mid, product_i=products[i], product_j=products[j], elasticity=E[i,j]))
        end
    end
    return out
end

true_own_elasticities = zeros(size(elasticities_df,1))
truth = Main.sim_true_price_elasticities(df_original, dropdims(β, dims=1), Σ)
own_elasticities = elasticities_df[elasticities_df.product1 .== elasticities_df.product2,:elast]

histogram(
    own_elasticities, 
    label = "Estimated", 
    xlabel = "Own-Price Elasticity", 
    ylabel = "Count", 
    color = :skyblue,
    alpha = 0.5
    )

histogram!(
    truth[truth.product_i .== truth.product_j,:elasticity], 
    label = "Truth", 
    color = :lightcoral, 
    alpha = 0.5
    )

scatter(
    own_elasticities, 
    truth[truth.product_i .== truth.product_j,:elasticity], 
    label = "", 
    color = :lightcoral, 
    alpha = 0.5,
    xlabel = "Estimated Own-Price Elasticity",
    ylabel = "True Own-Price Elasticity"
)
plot!(own_elasticities, own_elasticities, 
    label = "45 degree line", color = :black, ls = :dash, 
    lw = 3)

# Join the estimated elasticities with the true elasticities
# Plot against each other with 45 degree line
joined = leftjoin(
    elasticities_df, 
    rename(truth, 
        :product_i => :product1, 
        :product_j => :product2, 
        :elasticity => :truth),
    on = [:product1, :product2, :market_ids], 
    makeunique=true
) 
filter!(row -> row.product1 .== row.product2, joined)
# select!(joined, setdiff(Symbol.(names(joined)), [:product1, :product2, :market_ids])) 
rename!(joined,
    :truth => :true_own_elasticity, 
    :elast => :estimated_own_elasticity) 