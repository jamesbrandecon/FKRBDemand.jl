using FKRBDemand, Test, DataFrames

@testset "core functions" begin
    # logexpratio: zero inputs => equal shares
    x = reshape([0.0,0.0],2,1)
    β = [0.0]
    r = logexpratio(x, β)
    @test minimum(isapprox.(r, fill(1/3,2), atol=1e-8)) ==1

    # FKRBGridDetails holds ranges and method
    gd = FKRBGridDetails(Dict("a" => 1:1:3), "simple")
    @test gd.method == "simple"
    @test haskey(gd.ranges, "a")

    # FKRBProblem struct fields
    df = DataFrame(market_ids=[1], product_ids=[1])
    prob = FKRBProblem(
        df,                # data
        String[],          # linear
        String[],          # nonlinear
        String[],          # iv
        zeros(2,2),        # grid_points
        zeros(0,0),        # regressors
        Any[],             # results
        Int[],             # train
        Any[],             # inference_results
        Any[],             # std
        Any[]              # all_elasticities
        )
    @test prob.data === df
    @test prob.nonlinear == String[]
end
