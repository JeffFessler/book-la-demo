#=
# [Vector dot product](@id dot)

This example illustrates different ways of computing
[vector dot products](https://en.wikipedia.org/wiki/Dot_product)
using the Julia language.
=#

#srcURL

#=
First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "LinearAlgebra"
        "BenchmarkTools"
        "LazyGrids"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: dot
using BenchmarkTools: @benchmark
using InteractiveUtils: versioninfo
using LazyGrids: btime


#=
## Overview of dot products

The dot product between two vectors
is such a basic method in linear algebra
that, of course,
Julia has a function `dot` built-in for it.

In practice one should simply call that `dot` method.

This demo explores other ways of coding the dot product,
to illustrate,
in a simple setting,
techniques for writing efficient code.

We write each method as a function
because the most reliable way
to benchmark different methods
is to use functions.
=#

# ### The built-in `dot` method:
f1(x,y) = dot(y,x);

# ### An equivalent method using the adjoint `'`
# It can be written `y' * x` or `*(y', x)`.
# By checking `@which *(y', x)`
# one can verify that these all call `dot`.
f2(x,y) = y'x;

# ### Using `sum` with vector conjugate
# This is suboptimal because it must allocate memory for `conj(y)`
f3(x,y) = sum(conj(y) .* x); # must allocate "conj(y)"

# ### Using `zip` and `sum` with a function argument
# This approach avoids the needless allocation.
f4(x,y) = sum(z -> z[1] * conj(z[2]), zip(x,y));

# ### A basic `for` loop like one would write in a low-level language
function f5(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    for i in 1:length(x)
        accum += x[i] * conj(y[i])
    end
    return accum
end;

# ### An advanced `for` loop that uses bounds checking and SIMD operations
function f6(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    @boundscheck length(x) == length(y) || throw("incompatible")
    @simd for i in 1:length(x)
        @inbounds accum += x[i] * conj(y[i])
    end
    return accum
end;

# ### The Julia fallback method (from source code as of v1.8.1)
# This code is what is used for general `AbstractArray` types.
function f7(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    @boundscheck length(x) == length(y) || throw("incompatible")
    for (ix,iy) in zip(eachindex(x), eachindex(y))
        @inbounds accum += x[ix] * conj(y[iy]) # same as dot(y[iy], x[ix])
    end
    return accum
end;


# ### Data for timing tests
N = 2^16; x = rand(ComplexF32, N); y = rand(ComplexF32, N)

# Verify the methods are equivalent
@assert f1(x,y) == f2(x,y) ≈ f3(x,y) ≈ f4(x,y) ≈ f5(x,y) ≈ f6(x,y) ≈ f7(x,y)


# ## Benchmark the methods
# The results will depend on the computer used, of course.

# y'x
t = @benchmark f1($x,$y)
timeu = t -> btime(t, unit=:μs)
timeu(t)

# dot(y,x)
t = @benchmark f2($x,$y)
timeu(t)

# sum with conj()
t = @benchmark f3($x,$y)
timeu(t)

# zip sum
t = @benchmark f4($x,$y)
timeu(t)

# basic loop
t = @benchmark f5($x,$y)
timeu(t)

# fancy loop with @inbounds & @simd
t = @benchmark f6($x,$y)
timeu(t)

# zip accum loop
t = @benchmark f7($x,$y)
timeu(t)


#=
### Remarks

The built-in `dot` method is the fastest.
Behind the scenes it calls
[`BLAS.dot`](https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/blas.jl)
which is highly optimized
because it uses
[cpu specific assembly code](https://discourse.julialang.org/t/why-is-blas-dot-product-so-much-faster-than-julia-loop/44994)
based on
[Single instruction, multiple data (SIMD)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
to perform, say, 4 multiplies
in a single instruction.
Thus the basic loop is several times slower than `dot()`.

Sometimes we can speed up code
by promising the Julia compiler
that array indexing operations
like `x[i]` are valid,
by adding the `@inbounds` macro.

Depending on the CPU,
using `@simd` and `@inbounds`
can lead to speeds
close to that of `dot`.

The `promote_type` function ensures that the accumulator
uses the better precision of the two arguments.
=#


include("../../../inc/reproduce.jl")
