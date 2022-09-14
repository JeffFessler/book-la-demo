#---------------------------------------------------------
# # [Vector dot product](@id dot)
#---------------------------------------------------------

#=
This example illustrates different ways of computing
vector dot products
using the Julia language.
=#

#=
This entire page was generated using a single Julia file:
[dot.jl](@__REPO_ROOT_URL__/01/dot.jl).
=#
#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`svd-diff.ipynb`](@__NBVIEWER_ROOT_URL__/01/dot.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`svd-diff.ipynb`](@__BINDER_ROOT_URL__/01/dot.ipynb),

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
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: dot
using BenchmarkTools: @btime
using InteractiveUtils: versioninfo


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
f1(x,y) = dot(y,x)

# ### An equivalent method using the adjoint `'`
# It can be written `y' * x` or `*(y', x)`.
# By checking `@which *(y', x)`
# one can verify that these all call `dot`.
f2(x,y) = y'x

# ### Using `sum` with vector conjugate
# This is suboptimal because it must allocate memory for `conj(y)`
f3(x,y) = sum(conj(y) .* x) # must allocate "conj(y)"

# ### Using `zip` and `sum` with a function argument
# This approach avoids the needless allocation.
f4(x,y) = sum(z -> z[1] * conj(z[2]), zip(x,y))

# ### A basic `for` loop like one would write in a low-level language
function f5(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    for i in 1:length(x)
        accum += x[i] * conj(y[i])
    end
    return accum
end

# ### An advanced `for` loop that uses bounds checking and SIMD operations
function f6(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    @boundscheck length(x) == length(y) || throw("incompatible")
    @simd for i in 1:length(x)
        @inbounds accum += x[i] * conj(y[i])
    end
    return accum
end

# ### The Julia way (from source code as of v1.8.1)
function f7(x,y)
    accum = zero(promote_type(eltype(x), eltype(y)))
    @boundscheck length(x) == length(y) || throw("incompatible")
    for (ix,iy) in zip(eachindex(x), eachindex(y))
        @inbounds accum += x[ix] * conj(y[iy]) # same as dot(y[iy], x[ix])
    end
    return accum
end


# ### Data for timing tests
N = 2^16; x = rand(ComplexF32, N); y = rand(ComplexF32, N)

# Verify the methods are equivalent
@assert f1(x,y) == f2(x,y) ≈ f3(x,y) ≈ f4(x,y) ≈ f5(x,y) ≈ f6(x,y) ≈ f7(x,y)

# ## Benchmark the methods
# The results will depend on the computer used, of course.

#
@btime f1($x,$y); # y'x

#
@btime f2($x,$y); # dot(y,x)

#
@btime f3($x,$y); # sum with conj()

#
@btime f4($x,$y); # zip sum

#
@btime f5($x,$y); # basic loop

#
@btime f6($x,$y); # fancy loop (@inbounds & @simd may help)

#
@btime f7($x,$y); # zip accum loop


#=
### Observations:

The built-in `dot` method is the fastest.
Behind the scenes it calls `BLAS.dot`
which seems to be highly optimized.
I was hoping that using `@simd` and `@inbounds`
would lead to speeds
closer to `dot`.
=#


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()