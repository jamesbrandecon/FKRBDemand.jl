using FKRBDemand, Test, Random

@testset "elasticnet" begin
    # simple 1‐param regress: Y = 1 * X
    X = ones(10,2)
    Y = ones(10)
    b_en, b_ls = elasticnet(Y, X, 0.0, 0.0)
    @test isapprox(b_en[1], 0.5, atol=1e-6)
    @test isapprox(b_ls[1], 0.5, atol=1e-6)

    # L1 penalty pushes irrelevant terms towards zero
    Random.seed!(1)
    X2 = rand(1000,100)
    Y2 = rand(1000) + X2[:,1] * 0.5
    b_en, _ = elasticnet(Y2, X2, 0.0, 0.0)
    b_en2, _ = elasticnet(Y2, X2, 100.0, 0.0)
    @test (b_en2[2] <= b_en[2])
end
