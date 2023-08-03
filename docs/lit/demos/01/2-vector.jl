#=
# [Tutorial: Vectors in Julia](@id tutor-2-vector)

Vectors in Julia differ a bit from Matlab.  
In Matlab, everything is an array, including vectors (and even scalars).  
In Julia, there are distinct data types
for scalars, vectors, rowvectors, and 1D arrays.  
This notebook illustrates the differences.  
- Jeff Fessler, University of Michigan
- 2017-07-24, original
- 2020-08-05, Julia 1.5.0
- 2021-08-23, Julia 1.6.2
- 2023-08-03, Julia 1.9.2, Literate
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
        "InteractiveUtils"
        "LinearAlgebra"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

## using InteractiveUtils: versioninfo
## using LinearAlgebra:


#=
## Scalars, Vectors, Arrays
=#

a = 4 # this is a scalar

b1 = [4] # this is a Vector with one element

b2 = reshape([4], 1, 1) # here is a 1×1 Array

b3 = reshape([4], 1, 1, 1) # here is a 1×1×1 Array

# In Julia the following all differ! (In Matlab they are the same.)
a==b1, b1==b2, a==b2, b2==b3


#=
## Vectors and Transpose
=#

# This construction (with just spaces) makes a `1×3 Matrix`:
c = [4 5 6]

# This construction (using commas) makes a 1D `Vector`:
d = [4, 5, 6]

# So does this construction, whereas in Matlab the "," and ";" work differently:
e = [4; 5; 6]

# The transpose of a `Vector` is slightly different than a `1×N` array!
# This is a subtle point!
d'

# Nevertheless, the values are the same:
d' == c

# Transposing back gives a vector again (not a `N×1` array):
(d')'

# These are all true, as expected, despite the adjoint type:
d==e, d'==c, (c')'==d'

# These are all false:
c==d,  c==e,  c'==d,  (d')'==c'

# An "inner product" of a `1×3 Matrix` with a `3×1 Matrix`
# returns a `1×1 Matrix`, not a scalar:
c * c'

# This inner product of an `adjoint Matrix` with a `Vector` returns a scalar:
d' * d

# How to make a vector from an array:
vec(c)

# Here is another way (but it uses more memory than `vec`):
c[:]

#=
## Call by reference
Julia uses call-by-reference (not value), like C/C++, unlike Matlab!
=#

# Here `B` is the same "pointer" so this changes `A`:
A = zeros(2); B = A; B[1] = 7
A

# Here B is different, so this does not change `A`:
A = zeros(2); B = A .+ 2; B[1] = 7
A

# This changes `A` because `B` and `A` point to same data:
A = B = zeros(2); B[1] = 7
A

# This changes `B` for the same reason:
A = B = zeros(2); A[1] = 7
B

# To avoid this issue, one can use `copy`;

A = zeros(2); B = copy(A); B[1] = 7; # B here uses different memory than A
A # here it is unchanged

## include("../../../inc/reproduce.jl")
