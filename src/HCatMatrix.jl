export HCatMatrix

immutable HCatMatrix{T,S<:AbstractMatrix,V<:AbstractMatrix} <:AbstractMatrix{T}
    left::S
    right::V
end

function HCatMatrix(left::AbstractMatrix, right::AbstractMatrix)
    size(left, 1) == size(right, 1) ||
        error(DimensionMismatch("left and right must have same first dimension"))
    HCatMatrix{Base.promote_eltype(left, right),typeof(left),typeof(right)}(left, right)
end

Base.size(A::HCatMatrix) = (size(A.left, 1), size(A.left, 2)+size(A.right, 2))
Base.getindex(X::HCatMatrix, i1::Integer) =
    i1 <= length(X.left) ? X.left[i1] : X.right[i1-length(X.left)]
Base.getindex(X::HCatMatrix, i1::Integer, i2::Integer) =
    i2 <= size(X.left, 2) ? X.left[i1, i2] : X.right[i1, i2-size(X.left, 2)]

function Base.A_mul_B!(out::DenseVector, A::HCatMatrix, b::DenseVector)
    size(A, 1) == length(out) && size(A, 2) == length(b) ||
        error(DimensionMismatch(""))
    b1 = sub(b, 1:size(A.left, 2))
    b2 = sub(b, size(A.left, 2)+1:size(A.left, 2)+size(A.right, 2))
    if isa(A.left, DenseMatrix) && eltype(A.left) <: Base.LinAlg.BlasFloat
        A_mul_B!(out, A.right, b2)
        BLAS.gemv!('N', 1.0, A.left, b1, 1.0, out)
    else
        A_mul_B!(out, A.left, b1)
        broadcast!(+, out, out, A.right*b2)
    end
    out
end

for fn in (:Ac_mul_B, :At_mul_B)
    fn! = symbol(string(fn, '!'))
    @eval begin
        function Base.$fn!(out::DenseMatrix, A::HCatMatrix, B::HCatMatrix)
            A1 = A.left
            A2 = A.right
            B1 = B.left
            B2 = B.right
            size(out, 1) == size(A, 2) && size(A, 1) == size(B, 1) && size(out, 2) == size(B, 2) ||
                error(DimensionMismatch(""))
            $fn!(sub(out, 1:size(A1, 2), 1:size(B1, 2)), A1, B1)
            $fn!(sub(out, 1:size(A1, 2), size(B1, 2)+1:size(B1, 2)+size(B2, 2)), A1, B2)
            $fn!(sub(out, size(A1, 2)+1:size(A1, 2)+size(A2, 2), 1:size(B1, 2)), A2, B1)
            $fn!(sub(out, size(A1, 2)+1:size(A1, 2)+size(A2, 2),
                           size(B1, 2)+1:size(B1, 2)+size(B2, 2)), A2, B2)
            out
        end

        function Base.$fn!(out::DenseVector, A::HCatMatrix, b::DenseVector)
            length(out) == size(A, 2) && size(A, 1) == size(b, 1) || error(DimensionMismatch(""))
            A1 = A.left
            A2 = A.right
            $fn!(sub(out, 1:size(A1, 2)), A.left, b)
            $fn!(sub(out, size(A1, 2)+1:size(A1, 2)+size(A2, 2)), A.right, b)
            out
        end

        function Base.$fn!(out::DenseMatrix, A::HCatMatrix, B::DenseMatrix)
            size(out, 1) == size(A, 2) && size(A, 1) == size(B, 1) && size(out, 2) == size(B, 2) ||
                error(DimensionMismatch(""))
            A1 = A.left
            A2 = A.right
            $fn!(sub(out, 1:size(A1, 2), 1:size(B, 2)), A.left, B)
            $fn!(sub(out, size(A1, 2)+1:size(A1, 2)+size(A2, 2), 1:size(B, 2)), A.right, B)
            out
        end

        Base.$fn(X::HCatMatrix, Y::Union(HCatMatrix, DenseMatrix)) =
            $fn!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2), size(Y, 2)), X, Y)
        Base.$fn(X::HCatMatrix, Y::DenseVector) =
            $fn!(Array(promote_type(eltype(X), eltype(Y)), size(X, 2)), X, Y)
    end
end

function Base.scale!(out::HCatMatrix, s::DenseVector, X::HCatMatrix)
    scale!(out.left, s, X.left)
    scale!(out.right, s, X.right)
    out
end

Base.copy{T,S,V}(X::HCatMatrix{T,S,V}) = HCatMatrix{T,S,V}(copy(X.left), copy(X.right))
