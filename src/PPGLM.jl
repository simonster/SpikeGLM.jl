module PPGLM
using StatsBase, Reexport
export bsplines, raisedcosines
@reexport using GLM
include("KroneckerProduct.jl")
include("HCatMatrix.jl")
include("basis.jl")

type PPGLMPredChol{T,M<:AbstractMatrix,C} <: GLM.LinPred
    X::M                           # model matrix
    beta0::Vector{T}               # base vector for coefficients
    delbeta::Vector{T}             # coefficient increment
    scratchbeta::Vector{T}
    chol::C
end
function PPGLMPredChol{T}(X::AbstractMatrix{T})
    PPGLMPredChol(X, zeros(T, size(X, 2)), zeros(T, size(X, 2)), zeros(T, size(X, 2)), cholfact(X'X))
end

function GLM.delbeta!{T}(p::PPGLMPredChol{T}, r::Vector{T}, wt::Vector{T})
	cholfact!(weightmul!(p.chol.factors, p.X, wt))
	A_ldiv_B!(p.chol, At_mul_B!(p.delbeta, p.X, (r.*wt)))
	p
end

function StatsBase.fit{T<:FloatingPoint,V<:AbstractVector}(::Type{GeneralizedLinearModel},
                                                           X::AbstractMatrix{T}, y::V, d::UnivariateDistribution,
                                                           l::Link=canonicallink(d);
                                                           dofit::Bool=true,
                                                           wts::V=fill!(similar(y), one(eltype(y))),
                                                           offset::V=similar(y, 0), fitargs...)
    size(X, 1) == size(y, 1) || throw(DimensionMismatch("number of rows in X and y must match"))
    n = length(y)
    length(wts) == n || throw(DimensionMismatch("length(wts) does not match length(y)"))
    length(offset) == n || length(offset) == 0 || throw(DimensionMismatch("length(offset) does not match length(y)"))
    wts = T <: Float64 ? copy(wts) : convert(typeof(y), wts)
    off = T <: Float64 ? copy(offset) : convert(Vector{T}, offset)
    mu = GLM.mustart(d, y, wts)
    eta = GLM.linkfun!(l, similar(mu), mu)
    if !isempty(off)
        @inbounds @simd for i = 1:length(eta)
            eta[i] -= off[i]
        end
    end
    rr = GlmResp{typeof(y),typeof(d),typeof(l)}(y, d, l, eta, mu, offset, wts)
    res = GeneralizedLinearModel(rr, PPGLMPredChol(X), false)
    dofit ? fit(res; fitargs...) : res
end

end # module
