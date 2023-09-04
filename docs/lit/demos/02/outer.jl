#=
# [Vector outer product](@id outer)

This example illustrates different ways of computing
[vector outer products](https://en.wikipedia.org/wiki/Outer_product)
using the Julia language.
=#

#srcURL

#=
## Setup
Add the Julia packages used in this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "BenchmarkTools"
        "InteractiveUtils"
        "LazyGrids"
        "LinearAlgebra"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using BenchmarkTools: @benchmark
using InteractiveUtils: versioninfo
using LazyGrids: btime
using LinearAlgebra: mul!


#=
## Overview of outer products

The outer product between two vectors `x` and `y` is simply `x * y'`.

This demo explores a couple ways of coding that operation.

We write each method as a function
because the most reliable way
to benchmark different methods
is to use functions.

We are interested in the computation time,
not the time spent allocating memory,
so we use mutating `mul!` versions of all methods.
=#

# ### The built-in method:
f0!(out, x, y) = mul!(out, x, y');

# ### Hand-coded double loop
function f1!(out, x, y)
    for j in 1:length(y)
        @simd for i in 1:length(x)
            @inbounds out[i,j] = x[i] * conj(y[j])
        end
    end
    return out
end

# ### column times scalar
function f2!(out, x, y)
    for j in 1:length(y)
        @inbounds @. (@view out[:,j]) = x * conj(y[j])
    end
    return out
end

# ### Using threads across columns
function f3!(out, x, y)
    Threads.@threads for j in 1:length(y)
        @inbounds @. (@view out[:,j]) = x * conj(y[j])
    end
    return out
end

# ### Data for timing tests
M, N = 2^9, 2^10
T = ComplexF32
x = rand(T, M)
y = rand(T, N)
out = Matrix{T}(undef, M, N)
out2 = Matrix{T}(undef, M, N)

# Verify the methods are equivalent
@assert f0!(out,x,y) ≈ f1!(out2,x,y) # why is ≈ needed here?!
@assert f1!(out,x,y) == f2!(out2,x,y)
@assert f1!(out,x,y) == f3!(out2,x,y)


# ## Benchmark the methods
# The results will depend on the computer used, of course.

# x*y'
t = @benchmark f0!($out, $x, $y)
timeu = t -> btime(t, unit=:μs)
t0 = timeu(t)

# double loop
t = @benchmark f1!($out, $x, $y)
timeu = t -> btime(t, unit=:μs)
t1 = timeu(t)

# column times scalar
t = @benchmark f2!($out, $x, $y)
t2 = timeu(t)

# threads
t = @benchmark f3!($out, $x, $y)
t3 = timeu(t)

# Result summary:
["built-in" t0; "double" t1; "column" t2; "thread" t3]


#=
### Remarks

With Julia 1.9 a 2017 iMac with 8 threads
(4.2 GHz Quad-Core Intel i7),
the results are
* "time=235.7μs mem=0 alloc=0"
* "time=153.9μs mem=0 alloc=0"
* "time=153.0μs mem=0 alloc=0"
* "time=54.1μs mem=4096 alloc=43"

Interestingly,
the hand coded loop
is faster than the built-in `mul!` for `x * y'`.

The results in github's cloud may differ.
=#

include("../../../inc/reproduce.jl")
