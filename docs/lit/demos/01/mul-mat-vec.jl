#=
# [Matrix-vector product](@id mul-mat-vec)

This example illustrates different ways of computing
matrix-vector products
using the Julia language.
=#

#srcURL

#=
## Setup
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "BenchmarkTools"
        "InteractiveUtils"
    ])
end


# Tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using BenchmarkTools: @benchmark
using InteractiveUtils: versioninfo


#=
## Overview of matrix-vector multiplication

The product between a matrix and a compatible vector
is such a basic method in linear algebra
that, of course,
Julia has a function `*` built-in for it.

In practice one should simply call that method via
`A * x`
or possibly `*(A, x)`.

This demo explores other ways of coding the product,
to explore techniques for writing efficient code.

We write each method as a function
because the most reliable way
to benchmark different methods
is to use functions.
=#


# ## Built-in `*`
# Conventional high-level matrix-vector multiply function:
function mul0(A::Matrix, x::Vector)
    @boundscheck size(A,2) == length(x) || error("DimensionMismatch(A,x)")
    return A * x
end;


# ## Double loop over `m,n`
# This is the textbook version.

function mul_mn(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = similar(x, M)
    for m in 1:M
        inprod = zero(eltype(x)) # accumulator
        for n in 1:N
            inprod += A[m,n] * x[n]
        end
        y[m] = inprod
    end
    return y
end;

# Using `@inbounds`
function mul_mn_inbounds(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = similar(x, M)
    for m in 1:M
        inprod = zero(x[1]) # accumulator
        for n in 1:N
            @inbounds inprod += A[m,n] * x[n]
        end
        @inbounds y[m] = inprod
    end
    return y
end;

# ## Double loop over `n,m`
# We expect this way to be faster because of cache access over `m`.

function mul_nm(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = zeros(eltype(x), M)
    for n in 1:N
        for m in 1:M
            y[m] += A[m,n] * x[n]
        end
    end
    return y
end;

# With @inbounds
function mul_nm_inbounds(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = zeros(eltype(x), M)
    for n in 1:N
        for m in 1:M
            @inbounds y[m] += A[m,n] * x[n]
        end
    end
    return y
end;

# With `@inbounds` and `@simd`
function mul_nm_inbounds_simd(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = zeros(eltype(x), M)
    for n in 1:N
        @simd for m in 1:M
            @inbounds y[m] += A[m,n] * x[n]
        end
    end
    return y
end;

# And with `@views`
function mul_nm_inbounds_simd_views(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = zeros(eltype(x), M)
    for n in 1:N
        @simd for m in 1:M
            @inbounds @views y[m] += A[m,n] * x[n]
        end
    end
    return y
end;


# ## Row versions
# Loop over `m`.

function mul_row(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = similar(x, M)
    for m in 1:M
        y[m] = transpose(A[m,:]) * x
    end
    return y
end;

# with `@inbounds`
function mul_row_inbounds(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = similar(x, M)
    for m in 1:M
        @inbounds y[m] = transpose(A[m,:]) * x
    end
    return y
end;

# with `@views`
function mul_row_views(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = similar(x, M)
    for m in 1:M
        @views y[m] = transpose(A[m,:]) * x
    end
    return y
end;

# with both
function mul_row_inbounds_views(A::Matrix, x::Vector)
    (M,N) = size(A)
    @assert N == length(x)
    y = similar(x, M)
    for m in 1:M
        @inbounds @views y[m] = transpose(A[m,:]) * x
    end
    return y
end;


# ## Col versions
# Loop over `n`:

function mul_col(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = zeros(eltype(x), M)
    for n in 1:N
        y += A[:,n] * x[n]
    end
    return y
end;

# with broadcast via `@.` to coalesce operations:
function mul_col_dot(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = zeros(eltype(x), M)
    for n in 1:N
        @. y += A[:,n] * x[n]
    end
    return y
end;

# and with `@views`
function mul_col_dot_views(A::Matrix, x::Vector)
    (M,N) = size(A)
    y = zeros(eltype(x), M)
    for n in 1:N
        @views @. y += A[:,n] * x[n]
#src    @inbounds @views @. y += A[:,n] * x[n] # did not help
    end
    return y
end;


# ## Test and time each version
# The results will depend on the computer used, of course.

M = 2^11
N = M - 4 # non-square to stress test
A = randn(Float32, M, N)
x = randn(Float32, N);

flist = (mul0,
    mul_mn, mul_mn_inbounds,
    mul_nm, mul_nm_inbounds, mul_nm_inbounds_simd, mul_nm_inbounds_simd_views,
    mul_row, mul_row_inbounds, mul_row_views, mul_row_inbounds_views,
    mul_col, mul_col_dot, mul_col_dot_views,
);

for f in flist # warm-up and test each version
    @assert A * x ≈ f(A, x)
end;

out = Vector{String}(undef, length(flist))
for (i, f) in enumerate(flist) # benchmark timing for each
    b = @benchmark $f($A,$x)
    tim = round(minimum(b.times)/10^6, digits=1) # in ms
    tim = lpad(tim, 4)
    name = rpad(f, 27)
	alloc = lpad(b.allocs, 5)
	mem = round(b.memory/2^10, digits=1)
    tmp = "$name : $tim ms $alloc alloc $mem KiB"
    out[i] = tmp
    isinteractive() && println(tmp)
end
out


#=
The following results were for a
2017 iMac with 4.2GHz quad-core Intel Core i7
with macOS Mojave 10.14.6 and Julia 1.6.2.

As expected, simple `A*x` is the fastest,
but one can come quite close to that using proper double loop order
with `@inbounds` or using "dots" and `@views` to coalesce.
Without `@views` the vector versions have huge memory overhead!
=#

[
"mul0                       :  0.9 ms     1 alloc    16.1 KiB"
"mul_mn                     : 22.5 ms     1 alloc    16.1 KiB"
"mul_mn_inbounds            : 22.0 ms     1 alloc    16.1 KiB"
"mul_nm                     :  3.1 ms     1 alloc    16.1 KiB"
"mul_nm_inbounds            :  1.5 ms     1 alloc    16.1 KiB"
"mul_nm_inbounds_simd       :  1.5 ms     1 alloc    16.1 KiB"
"mul_nm_inbounds_simd_views :  1.5 ms     1 alloc    16.1 KiB"
"mul_row                    : 32.8 ms  2049 alloc 33040.1 KiB"
"mul_row_inbounds           : 32.7 ms  2049 alloc 33040.1 KiB"
"mul_row_views              : 22.4 ms     1 alloc    16.1 KiB"
"mul_row_inbounds_views     : 22.4 ms     1 alloc    16.1 KiB"
"mul_col                    : 16.0 ms  6133 alloc 98894.6 KiB"
"mul_col_dot                :  7.0 ms  2045 alloc 32975.6 KiB"
"mul_col_dot_views          :  1.5 ms     1 alloc    16.1 KiB"
];


include("../../../inc/reproduce.jl")
