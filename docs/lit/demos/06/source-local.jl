#=
# [Source localization](@id source-local)

This example illustrates source localization via
[multi-dimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling)
using the Julia language.
=#

#srcURL

#=
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "InteractiveUtils"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
        "Statistics"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: svd, norm, Diagonal
using MIRTjim: jim, prompt
using Plots: default, scatter, savefig
using Random: seed!
using Statistics: mean
default(); default(label="", markerstrokecolor=:auto, widen=true,
    guidefontsize=14, tickfontsize=12, legendfontsize=14)


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
D = [norm(C[:,j] - C[:,i]) for i in 1:J, j in 1:J] # "comprehension" in julia!
pd = jim(D, L"D", color=:cividis, xlabel=L"j", ylabel=L"i")

#src savefig(pd, "06_source_local1_d.pdf")


# ### MDS algorithm

# Compute Gram matrix by de-meaning squared distance matrix
S = D.^2 # squared distances
G = S .- mean(S, dims=1) # trick: use "broadcasting" feature of julia
G = G .- mean(G, dims=2) # now we have de-meaned the columns and rows of S
G = -1/2 * G;

# We still cannot determine visually the point locations:
pg = jim(G, L"G", color=:cividis, xlabel=L"j", ylabel=L"i")

#src savefig(pg, "06_source_local1_g.pdf")

# Examine singular values
(_, σ, V) = svd(G) # svd returns singular values in descending order
ps = scatter(σ, label="singular values (noiseless case)",
    xlabel=L"k", ylabel=L"σ_k") # two nonzero (d=2)

#
prompt()

#src savefig(ps, "06_source_local1_eig.pdf")


# ### Estimate the source locations using rank=2
Ch = Diagonal(sqrt.(σ[1:2])) * V[:,1:2]' # here is the key step

# ### Plot estimated source locations
pc = scatter(Ch[1,:], -Ch[2,:], xtick=-4:4, ytick=-3:3, aspect_ratio=1,
 title="Location estimates (noiseless case)")

#
prompt()


# ## Noisy case
seed!(0)
Dn = D + 0.3 * randn(size(D))
Sn = Dn.^2
Gn = Sn .- mean(Sn, dims=1)
Gn = Gn .- mean(Gn, dims=2) # de-meaned
Gn = -1/2 * Gn
pgn = jim(Gn, "G noisy", color=:cividis)


# Singular values
(_, sn, Vn) = svd(Gn)
psn = scatter(sn, label="singular values (noisy case)") # σ₂ ≫ σ₃

#
prompt()

# ### Plot estimated source locations from noisy distance measurements
Cn = Diagonal(sqrt.(sn[1:2])) * Vn[:,1:2]'
pcn = scatter(Cn[1,:], -Cn[2,:], xtick=-4:4, ytick=-3:3, aspect_ratio=1,
 title="Location estimates (noisy case)")

#
prompt()


# ## Constant bias case
seed!(0)
Db = D .+ 0.3 * maximum(D) # fairly large bias
Sb = Db.^2
Gb = Sb .- mean(Sb, dims=1)
Gb = Gb .- mean(Gb, dims=2) # de-meaned
Gb = -1/2 * Gb
pgb = jim(Gb, "G biased", color=:cividis)

# Singular values
(_, sb, Vb) = svd(Gb)
psb = scatter(sb, label="singular values (biased case)") # σ₂ ≫ σ₃

#
prompt()

# ### Plot estimated source locations from biased distance measurements
Cb = Diagonal(sqrt.(sb[1:2])) * Vb[:,1:2]'
pcb = scatter(Cb[1,:], -Cb[2,:], xtick=-4:4, ytick=-3:3, aspect_ratio=1,
 title="Location estimates (biased case)")

#
prompt()


# ## Equilateral triangle example
G = [-2 1 1; 1 -2 1; 1 1 -2] / (-6.)
(~, σ, V) = svd(G)
Ch = Diagonal(sqrt.(σ[1:2])) * V[:,1:2]'
scatter(Ch[1,:], Ch[2,:], aspect_ratio=1, title="Location estimates")

#
prompt()


include("../../../inc/reproduce.jl")
