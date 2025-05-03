function elasticities_df(problem::FKRBProblem)
    df_out = DataFrame(
        market_ids = Int[],
        product1   = eltype(problem.data.product_ids)[],
        product2   = eltype(problem.data.product_ids)[],
        elast      = Float64[]
    )

    for row in eachrow(problem.all_elasticities)
        m         = row.market_ids
        elast_mat = row.elasts
        # get the actual product IDs in market m
        prods     = unique(problem.data.product_ids[problem.data.market_ids .== m])
        J         = length(prods)

        for i in prods, j in prods
            e = elast_mat[i,j]
            # if !isnan(e)
                push!(df_out, (
                    market_ids = m,
                    product1   = i,
                    product2   = j,
                    elast      = e
                ))
            # end
        end
    end

    return df_out
end

function price_elasticities!(problem::FKRBProblem)
    try 
        @assert problem.results !=[];
    catch
        error("Results are empty -- problem must be estimated before price elasticities can be calculated")
    end

    # In FKRB, the RHS regressors are the "individual" level market shares, so 
    # we just need to re-generate them and use the estimated weights to average them 
    # call aggregate regressors function
    # s_i = generate_regressors_aggregate(problem, method = "level")
    s_i = problem.regressors;
    
    weights = problem.results["weights"];
    price_coefs = problem.grid_points[:,findfirst(problem.nonlinear .== "prices")];

    # df_out = DataFrame() # market_ids = unique(problem.data.market_ids)
    elast_vec = [];
    all_products = unique(problem.data.product_ids);
    Threads.@threads for m in unique(problem.data.market_ids)
        products_m = unique(problem.data[problem.data.market_ids .==m,:].product_ids);
        temp_m = zeros(length(all_products), length(all_products));
        temp_m .= NaN;
        for j1_ind in eachindex(all_products)
            if all_products[j1_ind] in products_m # if this product is in market m
                j1_ind_m = findfirst(products_m .== all_products[j1_ind]);
                j1 = products_m[j1_ind_m];
                # Grab row (all columns) corresponding to this product in this market
                s_i_j1 = s_i[(problem.data.market_ids .==m) .& (problem.data.product_ids .== j1),:];
                for j2_ind in eachindex(all_products)
                    if all_products[j2_ind] in products_m # if second product is in market m
                        j2_ind_m = findfirst(products_m .== all_products[j2_ind]);
                        j2 = products_m[j2_ind_m];
                        s_i_j2 = s_i[(problem.data.market_ids .==m) .& (problem.data.product_ids .== j2),:];
                        if j1==j2
                            ds_dp = 1 .* s_i_j1 .* (1 .- s_i_j1) .* price_coefs' * weights
                            ds_dp = ds_dp[1];
                        else
                            ds_dp = -1 .* s_i_j1 .* s_i_j2 .* price_coefs' * weights;
                            ds_dp = ds_dp[1];
                        end
                        price_j1 = problem.data[(problem.data.market_ids .==m) .& (problem.data.product_ids .== j1),:prices];
                        share_j2 = problem.data[(problem.data.market_ids .==m) .& (problem.data.product_ids .== j2),:shares];
                        es_ep = ds_dp * price_j1 ./ share_j2
                        temp_m[j1_ind,j2_ind] = es_ep[1];
                    end
                end
            end
        end
        push!(elast_vec, (m, temp_m))
    end

    df_out = sort(
        DataFrame(
            market_ids = getindex.(elast_vec,1), 
            elasts     = getindex.(elast_vec,2)
        ),
        :market_ids
        )

    problem.all_elasticities = df_out;
end

function price_elasticities_new!(problem::FKRBProblem)
    @assert !isempty(problem.results) "Must run estimate! before price_elasticities!"

    # 1) Extract regressors, weights, and coefficient grid
    R = Matrix(problem.regressors)              # N × Q
    w = problem.results["weights"]              # length-Q
    # find the column index for "prices" in nonlinear
    price_col = findfirst(isequal("prices"), problem.nonlinear)
    v_prices = problem.grid_points[:, price_col]  # Q-vector
    α = dot(v_prices, w)                         # scalar alpha

    # 2) Pull raw data columns into arrays
    df = problem.data
    m_ids = df.market_ids
    p_ids = df.product_ids
    P_vals = df.prices
    S_vals = df.shares

    # 3) Build grouping maps
    markets = unique(m_ids)                     # in original order
    Nmk = length(markets)
    # market → all row indices in that market
    m_to_rows = Dict(m => findall(==(m), m_ids) for m in markets)
    # global product list and mapping to matrix index
    all_prods = unique(p_ids)
    Pn = length(all_prods)

    # 4) Prepare result container
    mats = Vector{Matrix{Float64}}(undef, Nmk)

    # 5) Loop over markets
    for (mi, m) in enumerate(markets)
        rows = m_to_rows[m]
        # Which products appear here and map to their row
        prods_m  = p_ids[rows]
        row_by   = Dict(prods_m[i] => rows[i] for i in 1:length(rows))

        # Preallocate Pn×Pn matrix with NaN
        M_e = fill(NaN, Pn, Pn)

        # Loop over all global products for j1, j2
        for j1_ind in 1:Pn
            prod1 = all_prods[j1_ind]
            if haskey(row_by, prod1)
                r1 = row_by[prod1]
                # follow original: take only the first regressor entry
                s1 = R[r1, 1]
                for j2_ind in 1:Pn
                    prod2 = all_prods[j2_ind]
                    if haskey(row_by, prod2)
                        r2 = row_by[prod2]
                        s2 = R[r2, 1]
                        # derivative following original broadcast+indexing
                        ds_dp = if j1_ind == j2_ind
                            α * s1 * (1 - s1)
                        else
                            -α * s1 * s2
                        end
                        # compute elasticity
                        pj1  = P_vals[r1]
                        sh2  = S_vals[r2]
                        M_e[j1_ind, j2_ind] = ds_dp * pj1 / sh2
                    end
                end
            end
        end

        mats[mi] = M_e
    end

    # 6) Attach to problem exactly as before
    problem.all_elasticities = DataFrame(
        market_ids = markets,
        elasts     = mats
    )
    return nothing
end
