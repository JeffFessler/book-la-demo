#=
# [RMT and matrix completion](@id rmt-complete1)

This example examines noisy matrix completion
(estimating a low-rank matrix from noisy data with missing measurements)
through the lens of random matrix theory,
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

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, dot, norm, rank, svd, svdvals
using MIRTjim: prompt, jim
using Plots: default, gui, plot, plot!, scatter!, savefig
using Plots.PlotMeasures: px
using Random: seed!
using StatsBase: mean, var
default(markerstrokecolor=:auto, label="", widen=true, markersize = 6,
 labelfontsize = 24, legendfontsize = 18, tickfontsize = 14, linewidth = 3,
)
seed!(0)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);



#=
## Helper functions
=#

# Generate random data for one trial:
function gen1(
    θ::Real = 3,
    M::Int = 100,
    N::Int = 2M,
    p_obs::Real = 1, # probability an element is observed
    T::Type{<:Real} = Float32,
)
    mask = rand(M, N) .<= p_obs
    u = rand((-1,+1), M) / T(sqrt(M)) # Bernoulli just for variety
    v = rand((-1,+1), N) / T(sqrt(N))
    ## u = randn(T, M) / T(sqrt(M))
    ## v = randn(T, N) / T(sqrt(N))
    X = θ * u * v' # theoretically rank-1 matrix
    Z = randn(T, M, N) / T(sqrt(N)) # gaussian noise
    Y = mask .* (X + Z) # missing entries set to zero
    return Y, u, v, θ, p_obs
end;

# SVD results for 1 trial:
function trial1(args...)
    Y, u, v, θ, p_obs = gen1(args...)
    fac = svd(Y)
    σ1 = fac.S[1]
    u1 = fac.U[:,1]
    v1 = fac.Vt[1,:]
    return [σ1, abs2(dot(u1, u)), abs2(dot(v1, v))]
end;

# Average `nrep` trials:
trial2(nrep::Int, args...) = mean((_) -> trial1(args...), 1:nrep);


# SVD for each of multiple trials, for different SNRs and matrix sizes:
if !@isdefined(vgrid)

    ## Simulation parameters
    T = Float32
    p_obs = 0.49
    Mlist = [30, 300]
    θmax = 4
    nθ = θmax * 4 + 1
    nrep = 100
    θlist = T.(range(0, θmax, nθ));
    labels = map(n -> latexstring("\$M = $n\$"), Mlist)

    c = 0.7 # non-square matrix to test
    c4 = c^0.25
    tmp = ((θ, M) -> trial2(nrep, θ, M, ceil(Int, M/c) #= N =#, p_obs)).(θlist, Mlist')
    σgrid = map(x -> x[1], tmp)
    ugrid = map(x -> x[2], tmp)
    vgrid = map(x -> x[3], tmp)
end;


#=
## Results
Compare theory predictions and empirical results.
There is again notable agreement
between theory and empirical results here.
=#

# σ1 plot
colors = [:orange, :red]
θfine = range(0, 2θmax, 60θmax + 1)
θmod = θfine .* sqrt(p_obs) # key modification from RMT!
sbg(θ) = θ > c4 ? sqrt((1 + θ^2) * (c + θ^2)) / θ : 1 + √(c)
stheory = sbg.(θmod) * sqrt(p_obs) # note modification!
bm = s -> "\\mathbf{\\mathit{$s}}"
ylabel = latexstring("\$σ_1($(bm(:Y)))\$ (Avg)")
ps = plot(θfine, stheory, color=:blue, label="theory",
    aspect_ratio = 1, legend = :topleft,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (1,θmax), 1:θmax),
    annotate = (3.1, 3.6, latexstring("c = $c"), :left),
)
scatter!(θlist, σgrid[:,1], marker=:square, color=colors[1], label = labels[1])
scatter!(θlist, σgrid[:,2], marker=:circle, color=colors[2], label = labels[2])
plot!(θlist, θlist * p_obs, label=L"p \; θ", color=:black, linewidth=2,
    annotate = (3.1, 3.3, latexstring("p = $p_obs"), :left))

#
prompt()


# u1 plot
ubg(θ) = (θ > c4) ? 1 - c * (1 + θ^2) / (θ^2 * (θ^2 + c)) : 0
utheory = ubg.(θmod)
ylabel = latexstring("\$|⟨\\hat{$(bm(:u))}, $(bm(:u))⟩|^2\$ (Avg)")
pu = plot(θfine, utheory, color=:blue, label="theory",
    left_margin = 10px, legend = :bottomright,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (0,1), 0:0.5:1),
)
scatter!(θlist, ugrid[:,1], marker=:square, color=colors[1], label = labels[1])
scatter!(θlist, ugrid[:,2], marker=:circle, color=colors[2], label = labels[2])

#
prompt()


# v1 plot
vbg(θ) = (θ > c^0.25) ? 1 - (c + θ^2) / (θ^2 * (θ^2 + 1)) : 0
vtheory = vbg.(θmod)
ylabel = latexstring("\$|⟨\\hat{$(bm(:v))}, $(bm(:v))⟩|^2\$ (Avg)")
pv = plot(θfine, vtheory, color=:blue, label="theory",
    left_margin = 10px, legend = :bottomright,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (0,1), 0:0.5:1),
)
scatter!(θlist, vgrid[:,1], marker=:square, color=colors[1], label = labels[1])
scatter!(θlist, vgrid[:,2], marker=:circle, color=colors[2], label = labels[2])

#
prompt()


if false
 savefig(ps, "complete1-s.pdf")
 savefig(pu, "complete1-u.pdf")
 savefig(pv, "complete1-v.pdf")
 pp = plot(ps, pu, pv; layout=(3,1), size=(600, 900))
end


#=
## Image example

Apply an SVD-based matrix completion approach
to some noisy and incomplete image data.
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
title = latexstring("\$$(bm(:X))\$ : Latent image")
pt = jimc(Xtrue; title, xlabel = " ", args...)


#=
## Noisy / incomplete data
=#
seed!(0)
p_see = 0.8
mask = rand(Float32, size(Xtrue)) .<= p_see
sigZ = 6
Mx,Nx = sort(collect(size(Xtrue)))
Z = sigZ * randn(size(Xtrue)) # AWGN
Y = mask .* (Xtrue + Z);

title = latexstring("\$$(bm(:Y))\$ : Corrupted image matrix\n(missing pixels set to 0)")
py = jime(Y ; title)

# Show mask; count proportion of missing entries
frac_nonzero = count(mask) / length(mask)
title = latexstring("\$$(bm(:M))\$ : Locations of observed entries")
pm = jim1(mask; title, args...,
    xlabel = "sampled fraction = $(round(frac_nonzero * 100, digits=1))%")

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
c_4 = (Mx / Nx)^(1/4)
σcrit = sigZ^2 * sqrt(Nx) * c_4 / sqrt(p_see) # from RMT

pg = plot([1, Nx], [1, 1] * σcrit, color=:cyan,
 title="singular values",
 xaxis=(L"k", (1, Mx), [1, 3, 6, Mx]),
 yaxis=(L"σ_k",),
 leftmargin = 15px, bottommargin = 20px, size = (600,350), widen = true,
)
sv_x = svdvals(Xtrue)
sv_y = svdvals(Y)
scatter!(pg, sv_x, color=:blue, label="Xtrue", marker=:utriangle)
scatter!(pg, sv_y, color=:red, label="Y (data)", marker=:dtriangle)
scatter!(pg, sv_y[1:3] / p_see, color=:green, label="Y/p", marker=:hex, alpha=0.8)

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

#=
How well do the singular vectors match?
The first 3 components match quite well:
=#
[sum(svd(Xr).U[:,1:r] .* svd(Xtrue).U[:,1:r], dims=1).^2;
 sum(svd(Xr).V[:,1:r] .* svd(Xtrue).V[:,1:r], dims=1).^2]

# The next 2 components match very poorly, as predicted:
[sum(svd(Y).U[:,4:5]/p_see .* svd(Xtrue).U[:,4:5], dims=1).^2;
 sum(svd(Y).V[:,4:5]/p_see .* svd(Xtrue).V[:,4:5], dims=1).^2]

include("../../../inc/reproduce.jl")
