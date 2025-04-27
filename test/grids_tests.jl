using FKRBDemand, Test, DataFrames

@testset "grid generation" begin
    df = DataFrame(x=[1.0], prices=[2.0])
    gd = FKRBGridDetails(Dict("x" => 1:1:3, "prices" => 2:2:4), "simple")
    pts = make_grid_points(df, ["prices", "x"], ["prices", "x"]; gridspec=gd)
    # should produce 3 points for x × 2 points for prices = 6 rows
    @test size(pts) == (6,2)
    # first column = x-values, second = prices
    @test unique(pts[:,2]) == [1.0,2.0,3.0]
end
