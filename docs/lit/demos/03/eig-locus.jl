#=
# [Eigenvalue locus](@id eig-locus)

This example illustrates how the eigenvalues
of a ``2 × 2`` symmetric matrix evolve
with increasing off-diagonal components.
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
        "MIRTjim"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, det, eigvals
using MIRTjim: prompt
using Plots: default, plot, plot!, scatter!
default(); default(label="", markerstrokecolor=:auto, widen=:true)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt)


#=
## Eigenvalues
of ``(1 - c) A + c B``
for ``c ∈ [0,1]``
=#

A = Diagonal([8, 2])
B = [0 1; 1 0] # has eigenvalues ±1
N = 21
clist = range(0, 1, N)
d = det(A)
c0 = (d - sqrt(d)) / (d - 1) # c for which the convex combo becomes singular
clist = sort([clist; c0])
mats = [(1-c) * A + c * B for c in clist] # convex combinations
eigs = eigvals.(mats);

e1 = map(first, eigs)
e2 = map(last, eigs)
plot(xaxis = ("c", (0,1), 0:0.2:1), yaxis = ("λ"),)
plot!(clist, e2, label = "λ₂((1-c) A + c B)", color=:blue, width=2)
plot!(clist, e1, label = "λ₁((1-c) A + c B)", color=:red, width=2)
scatter!([0,0], eigvals(A), label="A eigs", color=:black)
scatter!([1,1], eigvals(B), label="B eigs", color=:black, marker=:square)
scatter!([c0], [0], color=:black, marker=:star)

# Add Gershgorin disk (intervals)
disk1u = [X[1] + X[3] for X in mats]
disk1d = [X[1] - X[3] for X in mats]
disk2u = [X[4] + X[2] for X in mats]
disk2d = [X[4] - X[2] for X in mats]
plot!(clist, disk1u, line=:dash, color=:blue)
plot!(clist, disk1d, line=:dash, color=:blue)
plot!(clist, disk2u, line=:dash, color=:red)
plot!(clist, disk2d, line=:dash, color=:red)

#
prompt()

include("../../../inc/reproduce.jl")
