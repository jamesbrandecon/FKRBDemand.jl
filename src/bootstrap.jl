
"""
    subsample!(problem::FKRBProblem)

# Arguments
- `problem::FKRBProblem`: FKRB problem (should be `estimated` first).
- `n::Int = nothing`: The number of samples to draw from the data. If `nothing`, a default value is used.
- `n_samples::Int = 100`: The number of bootstrap samples to generate.
- `lambda::Float64 = 1e-6`: The regularization strength parameter for the elastic net.

# Change to the problem object:
- `problem.inference_results`: A list of raw bootstrap results.
- `problem.std`: A vector of standard deviations for each weight parameter.
- `problem.results["boot_weights"]`: A matrix of weights, where each column corresponds to a subsample estimate.
"""
function subsample!(problem::FKRBProblem; 
    n = nothing,
    n_samples = 100, 
    lambda = 1e-6)

    df = problem.data;
    if n == nothing
        n = Int(floor(nrow(df)^(2/3))); # Rule of thumb
        start_string = "Subsampling with $(n_samples) replications using rule of thumb of n = $n ( = nrow(problem.data)^(2/3))"
    else
        start_string = "Subsampling with $(n_samples) replications using user-provided n = $n"
    end

    println(start_string)
    if (lambda ==1e-6)
        println("Using default penalty parameter lambda == $lambda")
    end
    
    results_store = [];
    for i in ProgressBar(1:n_samples)
        idn = StatsBase.sample(collect(1:nrow(df)), n, replace=false);
        df_sub =  df[idn, :];
        regressors_sub = problem.regressors[idn, :];
        sub_problem = FKRBProblem(df_sub, 
            problem.linear, 
            problem.nonlinear, 
            problem.iv, 
            problem.grid_points, 
            regressors_sub,
            problem.results, 
            problem.train,
            [],[],[]);
        estimate!(sub_problem, lambda = lambda)
        push!(results_store, sub_problem.results)
    end

    problem.inference_results = results_store;
    problem.std = [std(getindex.(getindex.(problem.inference_results,"weights"),i)) for i in 1:length(problem.results["weights"])];
    push!(
        problem.results, 
        "boot_weights" => hcat(getindex.(problem.inference_results,"weights")...)
    )
end


"""
    bootstrap!(problem::FKRBProblem; n_samples = 100, lambda = 1e-6, cross_validate = false)

Perform standard bootstrapping on the given `FKRBProblem` to estimate the uncertainty of model parameters and derived quantities.
This function modifies the `problem` object in place by adding bootstrap results to it.

# Arguments
- `problem::FKRBProblem`: FKRB problem (should be `estimated` first).
- `n_samples::Int = 100`: The number of bootstrap samples to generate.
- `lambda::Float64 = 1e-6`: The regularization strength parameter for the elastic net.
- `cross_validate::Bool = false`: If true, perform cross-validation to select the optimal penalty parameter.

# Change to the problem object: 
- `problem.inference_results`: A list of raw bootstrap results.
- `problem.std`: A vector of standard deviations for each weight parameter.
- `problem.results["boot_weights"]`: A matrix of bootstrap weights, where each column corresponds to a bootstrap sample.
"""
function bootstrap!(problem::FKRBProblem; n_samples = 100, 
    lambda = 1e-6, cross_validate = false)

    df = problem.data;
    start_string = "Starting boostrap with $(n_samples) replications..."
    println(start_string)

    results_store = [];
    for i in ProgressBar(1:n_samples)
        df_boot = df[sample(collect(1:nrow(df)), nrow(df), replace=true), :]
        boot_problem = FKRBProblem(df_boot, 
            problem.linear, 
            problem.nonlinear, 
            problem.iv, 
            problem.grid_points,
            zeros(2,2), # regressors
            problem.results, 
            problem.train, 
            [],[],[]);
        boot_problem.regressors = generate_regressors_aggregate(boot_problem, method = "level")
        estimate!(boot_problem, lambda = lambda, cross_validate = cross_validate)
        push!(results_store, boot_problem.results)
    end

    problem.inference_results = results_store;
    problem.std = [
        std(getindex.(getindex.(problem.inference_results,"weights"),i)) 
        for i in 1:length(problem.results["weights"])];
    push!(
        problem.results, 
        "boot_weights" => hcat(getindex.(problem.inference_results,"weights")...)
    )
end