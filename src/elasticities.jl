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

    df_out = DataFrame(market_ids = unique(problem.data.market_ids))
    elast_vec = [];
    all_products = unique(problem.data.product_ids);
    for m in unique(problem.data.market_ids)
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
        push!(elast_vec, temp_m)
    end
    df_out[!,"elasts"] = elast_vec;

    problem.all_elasticities = df_out;
end

function elasticities_df(problem::FKRBProblem)
    df_out = DataFrame(market_ids = [], product1 = [], product2 = [], elast = []);
    for m in unique(problem.data.market_ids)
        elast_mat = problem.all_elasticities[problem.all_elasticities.market_ids .==m,:].elasts[1];
        for j1 in unique(problem.data.product_ids)
            for j2 in unique(problem.data.product_ids)
                if !isnan(elast_mat[j1,j2])
                    push!(df_out, [m,j1,j2, elast_mat[j1,j2]])
                end
            end
        end
    end
    return df_out
end