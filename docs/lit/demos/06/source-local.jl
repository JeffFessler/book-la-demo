#=
# [Source localization](@id source-local)

This example illustrates source localization via
[multi-dimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling)
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
        "MIRTjim"
        "Plots"
        "Random"
        "Statistics"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: svd, norm, Diagonal
using Statistics: mean
using Random: seed!
using Plots; default(label="", markerstrokecolor=:auto)
#src using LaTeXStrings
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);
isinteractive() && jim(:prompt);


# ## Generate data

# Coordinates - can you identify the pattern?  probably not...
C0 = [
3 2.5 2 2 2 2 2 1 0 0 1 1 1 1 0 0 1 2 2.5 3  4 4.5 5 6 7 7 6 6 6 6 7 7 6 5 5 5 5 5 4.5 4;
2 3.0 4 3 2 1 0 0 0 1 1 2 3 4 4 5 5 5 4.0 3  3 4.0 5 5 5 4 4 3 2 1 1 0 0 0 1 2 3 4 3.0 2
];

# Interpolate the locations to make the final set more familiar
C2 = (C0[:,2:end] + C0[:,1:end-1])/2
Ce = (C0[:,1] + C0[:,end])/2
C3 = [C2 Ce]
C4 = [C0; C3]
C = reshape(C4, 2, :);

#src scatter(C[1,:], C[2,:], xtick=0:7, ytick=0:5) # check


# ### Compute J × J distance array and display it

J = size(C,2) # number of points
D = [norm(C[:,j] - C[:,i]) for i=1:J, j=1:J] # "comprehension" in julia!
jim(D, "D", color=:cividis)

#src savefig("06_source_local1_d.pdf")


# ### MDS algorithm

# Compute Gram matrix by de-meaning squared distance matrix
S = D.^2 # squared distances
G = S .- mean(S,dims=1) # trick: use "broadcasting" feature of julia
G = G .- mean(G,dims=2) # now we have de-meaned the columns and rows of S
G = -1/2 * G
jim(G, "G", color=:cividis) # still cannot determine visually the point locations

#src savefig("06_source_local1_g.pdf")

# Examine singular values
(_, σ, V) = svd(G) # svd returns singular values in descending order
scatter(σ, label="singular values") # two nonzero (d=2)

#
prompt()
#src savefig("06_source_local1_eig.pdf")


# ### Estimate the source locations using rank=2
Ch = Diagonal(sqrt.(σ[1:2])) * V[:,1:2]' # here is the key step

# ### Plot estimated source locations
scatter(Ch[1,:], -Ch[2,:], xtick=-4:4, ytick=-3:3, label="", aspect_ratio=1,
 title="Location estimates")

#
prompt()


# ## Noisy case
seed!(0)
Dn = D + 0.3 * randn(size(D))
Sn = Dn.^2
Gn = Sn .- mean(Sn,dims=1)
Gn = Gn .- mean(Gn,dims=2) # de-meaned
Gn = -1/2 * Gn
jim(Gn, "G noisy", color=:cividis) # still cannot determine visually the point locations


# Singular values
(_, sn, Vn) = svd(Gn)
scatter(abs.(sn), label="singular values") # two >> 0

#
prompt()

# ### Plot estimated source locations from noisy distance measurements
Cn = Diagonal(sqrt.(sn[1:2])) * Vn[:,1:2]' # here is the key step
scatter(Cn[1,:], -Cn[2,:], xtick=-4:4, ytick=-3:3, label="", aspect_ratio=1,
 title="Location estimates")

#
prompt()


# ## Equilateral triangle example
G = [-2 1 1; 1 -2 1; 1 1 -2] / (-6.)
(~, σ, V) = svd(G)
Ch = Diagonal(sqrt.(σ[1:2])) * V[:,1:2]'
scatter(Ch[1,:], Ch[2,:], aspect_ratio=1, label="",
 title="Location estimates")

#
prompt()


include("../../../inc/reproduce.jl")
