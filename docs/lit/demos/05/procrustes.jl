#=
# [Procrustes method](@id procrustes)

This example illustrates the
[orthogonal Procrustes method](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
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
        "InteractiveUtils"
        "LinearAlgebra"
        "MIRTjim"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LinearAlgebra: svd, norm, Diagonal
using MIRTjim: prompt


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Coordinate data

# coordinates from rotated image example in n-05-norm/fig/
A = [-59 -25 49;
    6 -33 20]
B = [-54.1 -5.15 32.44;
    -24.3 -41.08 41.82]

#src xc1 = [69, 103, 177] - 128
#src yc1 = 127 - [121, 160, 107]


# ## Procrustes solution steps

C = B * A'

#
(U,s,V) = svd(C)
s

#
Q = U * V'


# ## Fitting residual

residual = B - Q * A # small!


# Rotation angle in degrees:
acos(Q[1]) * (180/π) # very close to 30° as expected


# ## Fitting function

function procrustes(A, B)
    C = B * A'
    (U,s,V) = svd(C)
    Q = U*V'
    scale = sum(s) / norm(A,2)^2
    return Q, scale
end


# ## Explore additional special cases

# Three points along a line, symmetrical:

A = [-1 0 1; 0 0 0]
B = [0 0 0; -2 0 2]
Q, scale = procrustes(A, B)

# Check:
@assert B ≈ scale * Q * A


# Three points along a line, not symmetrical:

A = [-1 0 2; 0 0 0]
B = [0 0 0; -2 0 4]
Q, scale = procrustes(A, B)

# Check:
@assert B ≈ scale * Q * A


# A single point - works fine!

A = [1; 0]
B = [1; 1]
Q, scale = procrustes(A, B)

# Check:
@assert B ≈ scale * Q * A

# Angle:
acos(Q[1]) * 180/π


#src Q = U * Diagonal([1, 0]) * V'

#src todo: examine effect of noise too

#src prompt()


include("../../../inc/reproduce.jl")
