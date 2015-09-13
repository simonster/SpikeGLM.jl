export KroneckerProduct

typealias UnitStrideSubVector{T,N,P<:Array,I<:Tuple{Union{Colon,UnitRange{Int}},Vararg{Int}},LD} SubArray{T,N,P,I,LD}
typealias UnitStrideSubMatrix{T,N,P<:Array,I<:Tuple{Colon,Union{Colon,UnitRange{Int}},Vararg{Int}},LD} SubArray{T,N,P,I,LD}
typealias UnitStrideVector Union{Vector,UnitStrideSubVector}
typealias UnitStrideMatrix Union{Matrix,UnitStrideSubMatrix}

type KroneckerProduct{T,S<:AbstractMatrix,V<:AbstractMatrix,RA<:AbstractMatrix,RB<:AbstractMatrix} <: AbstractMatrix{T}
    A::S
    B::V
    multmp::Matrix{T}
    Arowtensor::RA
    Browtensor::RB

    KroneckerProduct(A, B) = new(A, B)
end
KroneckerProduct{S,V}(A::AbstractMatrix{S}, B::AbstractMatrix{V}) =
    KroneckerProduct{promote_type(S, V),typeof(A),typeof(B),rowtensor_type(A),rowtensor_type(B)}(A, B)

Base.size(X::KroneckerProduct) = (size(X.A, 1)*size(X.B, 1), size(X.A, 2)*size(X.B, 2))
Base.linearindexing{T<:KroneckerProduct}(::Type{T}) = Base.LinearSlow()

function Base.getindex(X::KroneckerProduct, i1::Int, i2::Int)
    blk1, j1 = divrem(i1-1, size(X.B, 1))
    blk2, j2 = divrem(i2-1, size(X.B, 2))
    X.A[blk1+1, blk2+1]*X.B[j1+1, j2+1]
end

Base.convert{T}(::Type{Matrix{T}}, X::KroneckerProduct) = convert(Matrix{T}, kron(X.A, X.B))
Base.convert{T}(::Type{Matrix}, X::KroneckerProduct{T}) = convert(Matrix{T}, X)

function multmp!{T}(X::KroneckerProduct{T})
    isdefined(X, :multmp) && return X.multmp
    X.multmp = Array(T, size(X.B, 1), size(X.A, 2))
end

function Base.A_mul_B!(out::UnitStrideVector, X::KroneckerProduct, v::AbstractVector)
    # v = (A ⊗ B)x <==> V = B*X*A'
    length(v) == size(X, 2) || throw(DimensionMismatch("X has size $(size(X)) but v has length $(length(v))"))
    A_mul_Bt!(pointer_to_array(pointer(out), (size(X.B, 1), size(X.A, 1))),
              A_mul_B!(multmp!(X), X.B, reshape(v, size(X.B, 2), size(X.A, 2))), X.A)
    out
end
function Base.A_mul_B!(out::StridedMatrix, X::KroneckerProduct, Y::AbstractMatrix)
    for i = 1:size(Y, 2)
        A_mul_B!(sub(out, :, i), X, sub(Y, :, i))
    end
    out
end

function Base.At_mul_B!(out::UnitStrideVector, X::KroneckerProduct, v::AbstractVector)
    # v = (A ⊗ B)'x <==> V = B'*X*A
    length(v) == size(X, 1) || throw(DimensionMismatch("X has size $(size(X)) but v has length $(length(v))"))
    At_mul_B!(pointer_to_array(pointer(out), (size(X.B, 2), size(X.A, 2))), X.B,
              A_mul_B!(multmp!(X), reshape(v, size(X.B, 1), size(X.A, 1)), X.A))
    out
end
function Base.At_mul_B!(out::StridedMatrix, X::KroneckerProduct, Y::AbstractMatrix)
    for i = 1:size(Y, 2)
        At_mul_B!(sub(out, :, i), X, sub(Y, :, i))
    end
    out
end

Base.At_mul_B(X::KroneckerProduct, Y::StridedMatrix) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2)), X, Y)
Base.At_mul_B(X::KroneckerProduct, Y::AbstractMatrix) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2)), X, Y)
Base.At_mul_B(X::KroneckerProduct, Y::AbstractVector) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2)), X, Y)
Base.Ac_mul_B{T<:Real}(X::KroneckerProduct{T}, Y::StridedMatrix) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2)), X, Y)
Base.Ac_mul_B{T<:Real}(X::KroneckerProduct{T}, Y::AbstractMatrix) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2)), X, Y)
Base.Ac_mul_B{T<:Real}(X::KroneckerProduct{T}, Y::AbstractVector) =
    At_mul_B!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2)), X, Y)

rowtensor_type{T}(::StridedMatrix{T}) = Matrix{T}
function rowtensor(X::StridedMatrix)
    out = zeros(eltype(X), size(X, 1), size(X, 2)^2)
    for k = 1:size(X, 2), j = 1:size(X, 2)
        @simd for i = 1:size(X, 1)
            @inbounds out[i, (k-1)*size(X, 2)+j] = X[i, j]*X[i, k]
        end
    end
    out
end

# Compute X'WX, where W = diagm(w)
function weightmul!(out::AbstractMatrix, X::AbstractMatrix, w::AbstractVector)
    # This would avoid a temporary, but BLAS is still substantially faster
#     for k = 1:size(X, 2), j = 1:k
#         v = zero(promote_type(eltype(X), eltype(w)))
#         @simd for i = 1:size(X, 1)
#             @inbounds v += X[i, j]*X[i, k]*w[i]
#         end
#         @inbounds out[j, k] = v
#     end
    Ac_mul_B!(out, scale(w, X), X)
end

function weightmul!(out::UnitStrideMatrix, X::KroneckerProduct, w::AbstractVector)
    if !isdefined(X, :Arowtensor)
        X.Arowtensor = rowtensor(X.A)
        X.Browtensor = rowtensor(X.B)
    end
    length(w) == size(X, 1) || throw(DimensionMismatch("X has size $(size(X)) but w has length $(length(w))"))
    tmp = A_mul_B!(similar(X, (size(X.B, 2)^2, size(X.A, 2)^2)),
             X.Browtensor'*reshape(w, size(X.B, 1), size(X.A, 1)), X.Arowtensor)
    permutedims!(pointer_to_array(pointer(out), (size(X.B, 2), size(X.A, 2), size(X.B, 2), size(X.A, 2))),
                 reshape(tmp, size(X.B, 2), size(X.B, 2), size(X.A, 2), size(X.A, 2)), (1, 3, 2, 4))
    out
end

weightmul{T,S}(X::AbstractMatrix{T}, w::AbstractVector{S}) =
    weightmul!(similar(X, promote_type(T, S), size(X, 2), size(X, 2)), X, w)
