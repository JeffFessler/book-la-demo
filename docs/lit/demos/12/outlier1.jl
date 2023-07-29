#=
# [RMT and outliers](@id outlier1)

This example examines the effects of outliers on SVD performance
for estimating a low-rank matrix from noisy data,
from the perspective of random matrix theory,
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
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
        "StatsBase"
    ])
end


# Tell Julia to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

#todo using Arpack
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, norm, rank, svd, svdvals
using MIRTjim: jim, prompt
using Plots.PlotMeasures: px
using Plots: default, gui, plot, plot!, scatter!, savefig
using Random: seed!
#todo using StatsBase: mean, var
default(markerstrokecolor=:auto, label="", widen=true, markersize = 6,
 labelfontsize = 24, legendfontsize = 18, tickfontsize = 14, linewidth = 3,
)
seed!(0)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

#todo isinteractive() && prompt(:prompt);



#=
## Image example

Apply an SVD-based low-rank approximation approach
to some data with outliers.
=#

#=
## Latent matrix
Make a matrix that has low rank:
=#
tmp = [
    zeros(1,20);
    0 1 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 1 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 1 0 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0;
    0 0 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0;
    zeros(1,20)
]';
rank(tmp)

# Turn it into an image:
Xtrue = kron(10 .+ 80*tmp, ones(9,9))
rtrue = rank(Xtrue)

# plots with consistent size
jim1 = (X ; kwargs...) -> jim(X; size = (700,300),
 leftmargin = 10px, rightmargin = 10px, kwargs...);
# and consistent display range
jimc = (X ; kwargs...) -> jim1(X; clim=(0,99), kwargs...);
# and with NRMSE label
nrmse = (Xh) -> round(norm(Xh - Xtrue) / norm(Xtrue) * 100, digits=1)
args = (xaxis = false, yaxis = false, colorbar = :none) # book
args = (;) # web
jime = (X; kwargs...) -> jimc(X; xlabel = "NRMSE = $(nrmse(X)) %",
 args..., kwargs...,
)
bm = s -> "\\mathbf{\\mathit{$s}}"
title = latexstring("\$$(bm(:X))\$ : Latent image")
pt = jimc(Xtrue; title, xlabel = " ", args...)


#=
## Helper functions
=#

# Bernoulli outliers with magnitude `τ` and probability `p`:
function outliers(dims::Dims, τ::Real = 60, p::Real = 0.05)
    Z = τ * sign.(randn(dims)) .* (rand(dims...) .< p)
    return Z
end


#=
## Noisy data
=#
seed!(0)
(M, N) = size(Xtrue)
Z = outliers((M,N))
Y = Xtrue + Z

title = latexstring("\$$(bm(:Y))\$ : Corrupted image matrix\n(with outliers)")
py = jime(Y ; title)

#=
## Singular values.

The first 3 singular values of ``Y``
are well above the "noise floor" caused by outliers.

But
``σ₄(X)``
todo
and ``σ₅(X)`` below the threshold,
is just barely above the threshold,
so we cannot expect a simple SVD approach
to recover them.
=#

ps = plot(
 title="singular values",
 xaxis=(L"k", (1, N), [1, 3, 6, N]),
 yaxis=(L"σ_k",),
 leftmargin = 15px, bottommargin = 20px, size = (600,350), widen = true,
)
sv_x = svdvals(Xtrue)
sv_y = svdvals(Y)
scatter!(ps, sv_y, color=:red, label="Y (data)", marker=:dtriangle)
scatter!(ps, sv_x, color=:blue, label="Xtrue", marker=:utriangle)

#
prompt()


#=
## Low-rank estimate

A simple low-rank estimate of ``X``
from the first few SVD components of ``Y``
works just so-so here.
A simple SVD approach recovers the first 3 components well,
but cannot estimate the 4th and 5th components.
=#

r = 5
U,s,V = svd(Y)
Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]'
title = latexstring("Rank $r approximation of data \$$(bm(:Y))\$")
pr = jime(Xr ; title)

savefig(pr, "tmp.pdf"); run(`open tmp.pdf`)
gui(); throw()


# Change outlier magnitude (tau) and probability of outlier (p).
# "Image with outliers (left) versus rank 4 reconstruction (right)"

function add(X, τ::Real = 10, p::Real = 0.001, r::Int = 5)
    Z = τ * sign.(randn(size(X))) .* (rand(size(X)) .< p)

    Y = X + Z
#   UsV = svds(Y |> real,nsv=4)[1]
    UsV = svd(Y)
    U = UsV.U[:,1:r]
    S = Diagonal(UsV.S[1:r])
    V = UsV.V[:,1:r]
    Xhat = U*S*V'
    return Y, Xhat
end
    return jim(stack((Xtrue, Y, Xhat)))

#@manipulate for τ in [0, 1, 10,100, 1000], p in [0, 0.001, 0.005, 0.05]
for τ in [10], p in [0.001]
end

pj = doit(10, 0.001)

#todo include("../../../inc/reproduce.jl")
