

#using Arpack
using LaTeXStrings
using LinearAlgebra: Diagonal, rank, svd
using MIRTjim: jim, prompt
using Plots.PlotMeasures: px
using Random: seed!

prompt(:prompt)

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

#=
## Noisy / incomplete data
=#
seed!(0)
# Mx,Nx = sort(collect(size(Xtrue)))
# Y = Xtrue + Z

title = latexstring("\$$(bm(:Y))\$ : Corrupted image matrix\n(with outliers)")
py = jime(Y ; title)


#=
## Singular values.

The first 3 singular values of ``Y``
are well above the "noise floor" caused by masking,
but, relative to those of ``X``
they are scaled down by a factor of ``p`` as expected.

We also show the critical value of ``σ``
where the phase transition occurs.
``σ₄(X)`` is just barely above the threshold,
and ``σ₅(X)`` below the threshold,
so we cannot expect a simple SVD approach
to recover them.
=#

pg = plot(
 title="singular values",
 xaxis=(L"k", (1, Mx), [1, 3, 6, Mx]),
 yaxis=(L"σ_k",),
 leftmargin = 15px, bottommargin = 20px, size = (600,350), widen = true,
)
sv_x = svdvals(Xtrue)
sv_y = svdvals(Y)
scatter!(pg, sv_x, color=:blue, label="Xtrue", marker=:utriangle)
scatter!(pg, sv_y, color=:red, label="Y (data)", marker=:dtriangle)

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
r = 3
U,s,V = svd(Y)
s ./= p_see # correction for masking effect
Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]'
title = latexstring("Rank $r approximation of data \$$(bm(:Y))\$")
pr = jime(Xr ; title)


=#


# Change outlier magnitude (tau) and probability of outlier (p).
# "Image with outliers (left) versus rank 4 reconstruction (right)"

function doit(τ, p)
    siz = size(Xtrue)
    Z = τ * sign.(randn(siz)) .* (rand(siz) .< p)
    Y = Xtrue + Z 
#   UsV = svds(Y |> real,nsv=4)[1]
    UsV = svd(Y)
    r = 4
    U = UsV.U[:,1:r]
    S = Diagonal(UsV.S[1:r])
    V = UsV.V[:,1:r]
    Xhat = U*S*V'
    return jim(stack((Xtrue, Y, Xhat)))
end

#@manipulate for τ in [0, 1, 10,100, 1000], p in [0, 0.001, 0.005, 0.05]
for τ in [10], p in [0.001]
end

pj = doit(10, 0.001)

#todo include("../../../inc/reproduce.jl")
