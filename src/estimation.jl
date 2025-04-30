""" 
    elasticnet(Y, X, γ, λ = 0)
Performs elastic net regularization for linear regression. 
Code modified from Convex.jl documentation: https://jump.dev/Convex.jl/stable/examples/general_examples/lasso_regression/
"""
function elasticnet(
        Y, X, 
        γ = 0.5, # weight between L1 and L2
        λ = 1e-6; 
        silent = true, 
        constraints = nothing)

    (T, K) = (size(X, 1), size(X, 2))
    X = Matrix(X);
    Y = Vector(Y);

    # println("Solving OLS problem...")
    # b_ls = X \ Y                    #LS estimate of weights, no restrictions

    Q = X'X / T
    c = X'Y / T                      #c'b = Y'X*b

    b = Variable(K)              #define variables to optimize over
    L1 = quadform(b, Q)            #b'Q*b
    L2 = dot(c, b)                 #c'b
    L3 = norm(b, 1)                #sum(|b|)
    L4 = sumsquares(b)/2            #sum(b^2)

    # define constraints 
    if (!isnothing(constraints)) && (:nonnegativity in constraints)
        constraints = [b >= 0];
    else
        constraints = [];
    end 

    if :proper_weights in constraints
        constraints = [constraints; sum(b)==1]
    end

    if λ > 0
        Sol = minimize(L1 - 2 * L2 + λ * (γ * L3 + (1-γ) * L4), constraints...)      #u'u/T + γ*sum(|b|) + λ*sum(b^2), where u = Y-Xb
    else
        Sol = minimize(L1 - 2 * L2 + γ * L3, constraints...)               #u'u/T + γ*sum(|b|) where u = Y-Xb
    end
    if !silent
        println("Solving elastic net problem...")
        println("λ = ", λ, ", γ = ", γ)
    end
    solve!(Sol, SCS.Optimizer; silent = silent)
    Sol.status == Convex.MOI.OPTIMAL ? b_i = vec(Convex.evaluate(b)) : b_i = NaN

    return b_i, zeros(K) #b_ls
end

function elastic_net_cv(X, Y, γ, λ, K;
    silent = true,
    constraints = nothing)

    n = size(X, 1)
    fold_size = n ÷ K
    errors = Float64[]

    for k in 1:K
        idx = ((k-1)*fold_size + 1):min(k*fold_size, n)
        test_idx = falses(n)
        test_idx[idx] .= true
        train_idx = .!test_idx

        X_train, Y_train = X[train_idx, :], Y[train_idx]
        X_test, Y_test   = X[test_idx, :],   Y[test_idx]

        β, _ = elasticnet(Y_train, X_train, γ, λ, silent = silent, constraints = constraints)
        predictions = X_test * β
        error = mean((Y_test - predictions).^2)
        push!(errors, error)
    end

    mean_error = mean(errors)
    return mean_error
end
