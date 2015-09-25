using PPGLM
using Base.Test

# write your own tests here
A = randn(100, 5)
B = randn(100, 3)
C = HCatMatrix(A, B)
D = [A B]
E = randn(100, 4)

a = randn(100)
b = randn(8)

@test_approx_eq C*b D*b
@test_approx_eq C'C D'D
@test_approx_eq C'E D'E
@test_approx_eq C'a D'a
@test_approx_eq C.'C D.'D
@test_approx_eq C.'E D.'E
@test_approx_eq C.'a D.'a
@test_approx_eq scale!(a, C) scale!(a, D)

A = randn(3, 2)
B = randn(5, 7)
densekron = kron(A, B)
kronprod = KroneckerProduct(A, B)
@test_approx_eq densekron PPGLM.kron!(similar(densekron), A, B)
@test_approx_eq densekron kronprod
@test_approx_eq densekron convert(Matrix, kronprod)

@test_approx_eq densekron'densekron kronprod'kronprod

v = randn(14)
@test_approx_eq densekron*v kronprod*v
Y = randn(14, 3)
@test_approx_eq densekron*Y kronprod*Y

v = randn(15)
@test_approx_eq densekron'v kronprod'v
@test_approx_eq v'densekron v'kronprod
Y = randn(15, 3)
@test_approx_eq densekron'Y kronprod'Y

@test_approx_eq kron(A, ones(1, size(A, 2))).*kron(ones(1, size(A, 2)), A) PPGLM.rowtensor(A)
@test_approx_eq kron(B, ones(1, size(B, 2))).*kron(ones(1, size(B, 2)), B) PPGLM.rowtensor(B)

@test_approx_eq PPGLM.weightmul(densekron, v) PPGLM.weightmul(kronprod, v)
@test_approx_eq PPGLM.weightmul([densekron Y], v) PPGLM.weightmul(HCatMatrix(kronprod, Y), v)

A = randn(10, 7)
B = randn(10, 10)
Y = randn(100, 3)
v = randn(100)
c1 = coef(fit(GeneralizedLinearModel, [kron(A, B) Y], v, Normal(), IdentityLink()))
c2 = coef(fit(GeneralizedLinearModel, HCatMatrix(KroneckerProduct(A, B), Y), v, Normal(), IdentityLink()))

const DATADIR = joinpath(dirname(@__FILE__), "data")

#= in R
	library(splines)
	x <- bs(seq(0, 1000), knots=seq(100, 900, 100), degree=3)
	write.table(x, "splines.csv", row.names=F, col.names=F)
=#
@test_approx_eq bsplines(0:1000, 3, 0:100:1000) readdlm(joinpath(DATADIR, "splines.txt"))

#= with neuroGLM
    bs = basisFactory.makeSmoothTemporalBasis('raised cosine', 1000, 20, @(x)x);
    x = bs.B
    save -ascii -double raisedcosines.txt x
 =#
@test_approx_eq raisedcosines(1:1000, 20) readdlm(joinpath(DATADIR, "raisedcosines.txt"))

#= with neuroGLM
    bs = basisFactory.makeNonlinearRaisedCos(10, 1, [0 100], 2);
    x = bs.B
    save -ascii -double raisedcosines.txt x
 =#
@test_approx_eq raisedcosines(0:242, 10, 0, 100, 2) readdlm(joinpath(DATADIR, "nonlinear_raisedcosines.txt"))
