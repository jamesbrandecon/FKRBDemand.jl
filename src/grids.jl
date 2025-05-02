""" 
    make_grid_points(data, linear, nonlinear)

Generates a grid of points for evaluating the FKRB model. Currently a very simple implementation, though easy to 
extend. 

# # Arguments
- `data::DataFrame`
- `linear::Vector{String}`: A vector of variable names from `data` indicating the terms in the utility function.
- `nonlinear::Vector{String}`: A vector of variable names from `data`. Should match `linear`-- will eventually collapse these into one input.

# Output: 
- `grid_points::Matrix{Float64}`: A matrix of grid points for the linear/nonlinear variables. Each row corresponds to a grid point, and each column corresponds to a variable.
"""
function make_grid_points(data, linear, nonlinear; gridspec::FKRBGridDetails)
    all_vars = union(linear, nonlinear);

    # Unpack grid details 
    range_dict = gridspec.ranges
    method = gridspec.method

    if method == "simple"
        store_univariates = [];
        for var in nonlinear
            univariate_grid = collect(range_dict[var])
            push!(store_univariates, univariate_grid)
        end
        grid_points = collect(Iterators.product(store_univariates...))
        grid_points = Matrix(reduce(hcat, map(collect, grid_points))'); # Final Matrix() wrap is to remove adjoint
    else
        throw(ArgumentError("Method not recognized"))
    end

    return grid_points
end