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
    w = problem.results[1];
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
    w = problem.results[1];
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