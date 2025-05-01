# visualization

"""
    plot_cdfs(problem::FKRBProblem)
Plots the estimated CDFs of the coefficients, along with the true CDFs (set manually for each simulation).
"""
function plot_cdfs(problem::FKRBProblem)
    # Unpack 
    data = problem.data
    linear = problem.linear;
    nonlinear = problem.nonlinear;
    all_vars = union(linear, nonlinear);
    w = problem.results["weights"];
    grid_points = problem.grid_points;

    # Calculate the CDFs
    cdf_plot = plot(xlabel = "β", ylabel = "P(b<=β)", legend = :outerright)
    for nl in eachindex(nonlinear) 
        unique_grid_points = sort(unique(grid_points[:,nl]));
        cdf_nl = [sum(w[findall(grid_points[:,nl] .<= unique_grid_points[i])]) for i in 1:size(unique_grid_points, 1)] 
        plot!(cdf_plot, unique_grid_points, cdf_nl, label = "Est. CDF $(nl)", lw = 1.2)
    end
    return cdf_plot
end

""" 
    plot_pdfs(problem::FKRBProblem)

Plots the estimated PDFs of the coefficients, along with the true PDFs (set manually for each simulation).
"""
function plot_pmfs(problem::FKRBProblem)
    # Unpack 
    data = problem.data
    linear = problem.linear;
    nonlinear = problem.nonlinear;
    all_vars = union(linear, nonlinear);
    w = problem.results["weights"];
    grid_points = problem.grid_points;

    # Calculate and plot the PMFs
    pmf_plot = plot(xlabel = "β", ylabel = "P(b=β)", legend = :outerright)
    for nl in eachindex(nonlinear) 
        unique_grid_points = sort(unique(grid_points[:,nl]));
        pdf_nl = [sum(w[findall(grid_points[:,nl] .== unique_grid_points[i])]) for i in 1:size(unique_grid_points, 1)] 
        plot!(pmf_plot, unique_grid_points, pdf_nl, label = "Est. PDF $(nl)", lw = 1.2)
    end
    return pmf_plot
end

function plot_joint_heatmap(grid_points::AbstractMatrix,
    weights::AbstractVector,
    dim1::Int, dim2::Int;
    normalize::Bool = true,
    kwargs...)
    # normalize
    w = normalize ? weights ./ sum(weights) : weights

    # extract and sort unique coords
    xs = sort(unique(grid_points[:, dim1]))
    ys = sort(unique(grid_points[:, dim2]))

    # prepare Z matrix: rows→ys, cols→xs
    Z = zeros(length(ys), length(xs))
    for (gp, wi) in zip(eachrow(grid_points), w)
    xi = findfirst(==(gp[dim1]), xs)
    yi = findfirst(==(gp[dim2]), ys)
    Z[yi, xi] += wi
    end

    heatmap(xs, ys, Z;
    xlabel = "dim $dim1",
    ylabel = "dim $dim2",
    colorbar = true,
    kwargs...)
end

import Plots.plot

"""
    plot_coefficients(
        problem::FKRBProblem;
        select_dims::Vector{String}=dim_names,
        heatmap_kwargs   = (c = cgrad([:white, :lightblue, :blue]), alpha = 0.6),
        marg_kwargs      = (c = :lightblue, alpha = 0.6),
    )

Create a pairs matrix of weighted marginal histograms (on the diagonal) and joint heatmaps
(off–diagonal lower triangle), styled like an R “pairs()” summary.  You can select a subset
of dimensions to plot via `select_dims`.

# Arguments
- `select_dims::Vector{String}`: subset of `dim_names` to include (default: all).
- `heatmap_kwargs`: NamedTuple of keyword args passed to `heatmap()`.
- `marg_kwargs`: NamedTuple of keyword args passed to `histogram()`.

# Returns
A Plots.jl plot object with an m×m grid, where m = length(select_dims).
"""
function plot_coefficients(
        problem;
        select_dims::Vector{String}= problem.nonlinear,
        heatmap_kwargs   = (c = cgrad([:white, :lightblue, :blue]), alpha = 0.6),
        marg_kwargs      = (c = :lightcoral, fillrange = 0, alpha = 0.5, nbins=50, fillalpha = 0.1),
        include_CI::Bool = false
    )   

    # Unpack
    grid_points = problem.grid_points;
    w           = problem.results["weights"];
    dim_names   = problem.nonlinear;

    # Map each name to its column index
    name_to_idx = Dict(name => i for (i, name) in enumerate(dim_names))
    # Look up indices for the selected dimensions (error if invalid)
    idxs = [ getindex(name_to_idx, nm) for nm in select_dims ]

    # Subset the grid and names
    sub_grid  = grid_points[:, idxs]
    sub_names = select_dims

    m = size(sub_grid, 2)           # number of selected dims
    panels = Vector{Plots.Plot}(undef, m * m)

    for i in 1:m, j in 1:m
        idx = (i - 1) * m + j

        # axis labels only on left column and bottom row
        xlabel = (i == m ? sub_names[j] : "")
        ylabel = (j == 1 ? sub_names[i] : "")
        title  = (i == 1 ? sub_names[j] : "")

        if i == j
            # Diagonal: weighted marginal histogram
            vals = sub_grid[:, i]
            kde_est = kde(problem.grid_points[:,i]; weights = problem.results["weights"]);
            base_plot = plot(
                    kde_est.x, kde_est.density;
                    legend  = false,
                    xlabel  = xlabel,
                    ylabel  = ylabel,
                    title   = title,
                    marg_kwargs...,
                )
        
            # if problem.results has a "boot_weights" key, estimate a histogram on the same edges
            # for each column, then add as CIs to the plot
            if include_CI && haskey(problem.results, "boot_weights")
                boot_weights = problem.results["boot_weights"]
                for bi in 1:size(boot_weights, 2)
                    kde_boot = kde(problem.grid_points[:,i], weights = boot_weights[:, bi]) 
                    # add CIs to the plot
                    plot!(
                        kde_boot.x, kde_boot.density;
                        label = false,
                        color = :grey,
                        alpha = 0.2
                    )
                end
            end
                
            panels[idx] = base_plot

        elseif i > j
            # Lower triangle: joint heatmap
            xs = unique(sub_grid[:, j]) |> sort
            ys = unique(sub_grid[:, i]) |> sort

            # aggregate weights into a matrix
            W = zeros(length(ys), length(xs))
            for q in 1:length(w)
                xv = sub_grid[q, j]
                yv = sub_grid[q, i]
                ix = searchsortedfirst(xs, xv)
                iy = searchsortedfirst(ys, yv)
                W[iy, ix] += w[q]
            end

            panels[idx] = heatmap(
                xs, ys, W;
                xlabel  = xlabel,
                ylabel  = ylabel,
                title   = title,
                legend  = false,
                heatmap_kwargs...,
            )

        else
            # Upper triangle: empty
            panels[idx] = plot(
                framestyle = :none,
                xlabel     = "",
                ylabel     = "",
                title      = title,
                xticks     = false,
                yticks     = false,
            )
        end
    end

    return plot(
        panels...;
        layout = (m, m),
        link   = :none,
        size   = (300 * m, 300 * m),
    )
end
