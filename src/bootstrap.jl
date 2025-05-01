# bootstrap 
function subsample!(problem::FKRBProblem; 
    n = nothing,
    n_samples = 100, 
    gamma = 0.001,
    lambda = 1e-6)

    df = problem.data;
    if n == nothing
        n = Int(floor(nrow(df)^(2/3))); # Rule of thumb
        start_string = "Subsampling with $(n_samples) replications using rule of thumb of n = $n ( = nrow(problem.data)^(2/3))"
    else
        start_string = "Subsampling with $(n_samples) replications using user-provided n = $n"
    end

    println(start_string)
    if ((gamma == 0.001)  & (lambda ==1e-6))
        println("Using default penalty parameters gamma == $gamma and lambda == $lambda")
    end
    
    results_store = [];
    for i in 1:n_samples
        df_sub =  df[StatsBase.sample(collect(1:nrow(df)), n, replace=false), :];
        sub_problem = FKRBProblem(df_sub, 
            problem.linear, 
            problem.nonlinear, 
            problem.iv, 
            problem.grid_points, 
            problem.results, problem.train,
            [],[],[]);
        estimate!(sub_problem, gamma = gamma, lambda = lambda)
        push!(results_store, sub_problem.results)
    end

    problem.inference_results = results_store;
    problem.std = [std(getindex.(getindex.(problem.inference_results,1),i)) for i in 1:length(problem.results["weights"])]
end


function bootstrap!(problem::FKRBProblem; n_samples = 100, 
    lambda = 1e-6, cross_validate = false)

    df = problem.data;
    start_string = "Starting boostrap with $(n_samples) replications..."
    println(start_string)

    results_store = [];
    for i in 1:n_samples
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

    problem.std = [std(getindex.(getindex.(problem.inference_results,1),i)) for i in 1:length(problem.results["weights"])]
end