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

A = randn(100, 7)
B = randn(100, 10)
Y = randn(10000, 3)
v = randn(10000)
c1 = coef(fit(GeneralizedLinearModel, [kron(A, B) Y], v, Normal(), IdentityLink()))
c2 = coef(fit(GeneralizedLinearModel, HCatMatrix(KroneckerProduct(A, B), Y), v, Normal(), IdentityLink()))
