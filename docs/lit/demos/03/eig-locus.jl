#=
# [Eigenvalue locus](@id eig-locus)

This example illustrates how the eigenvalues
of a ``2 × 2`` symmetric matrix evolve.
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
        "LaTeXStrings"
        "LinearAlgebra"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, eigvals
using MIRTjim: prompt
using Plots: default, plot, plot!
default(); default(label="", markerstrokecolor=:auto, widen=:true)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt)


#=
## Eigenvalues
of ``(1 - c) A + c B``
for ``c ∈ [0,1]``
=#

A = Diagonal([3, 2])
B = [0 1; 1 0]
N = 101
clist = range(0, 1, N)
eigs = [eigvals((1-c) * A + c * B) for c in clist];

e1 = map(x -> x[1], eigs)
e2 = map(x -> x[2], eigs)
plot(xaxis = ("c"), yaxis = ("λ"),)
plot!(clist, e2, label = "λ₂(A + c B)", color=:blue)
plot!(clist, e1, label = "λ₁(A + c B)", color=:red)

#
prompt()

include("../../../inc/reproduce.jl")
