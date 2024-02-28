#=
# [Random matrix theory and rank-1 signal + noise](@id rmt-gauss1)

This example compares results from random matrix theory
with empirical results
for rank-1 matrices with additive white Gaussian noise
using the Julia language.
This demo illustrates the phase transition
that occurs when the singular value
is sufficiently large
relative to the matrix aspect ratio ``c = M/N``.
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
using LinearAlgebra: dot, rank, svd, svdvals
using MIRTjim: prompt, jim
using Plots: default, gui, plot, plot!, scatter!, savefig, histogram
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
    T::Type{<:Real} = Float32,
)
    u = randn(T, M) / T(sqrt(M))
    v = randn(T, N) / T(sqrt(N))
    X = θ * u * v' # theoretically rank-1 matrix
    Z = randn(T, M, N) / T(sqrt(N)) # gaussian noise
    Y = X + Z
    return Y, u, v, θ
end;

# SVD results for 1 trial:
function trial1(args...)
    Y, u, v, θ = gen1(args...)
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
    Mlist = [30, 300]
    θmax = 4
    nθ = θmax * 4 + 1
    nrep = 100
    θlist = T.(range(0, θmax, nθ));
    labels = map(n -> latexstring("\$M = $n\$"), Mlist)

    c = 1 # square matrices for simplicity
    c4 = c^0.25
    tmp = ((θ, M) -> trial2(nrep, θ, M, ceil(Int, M/c) #= N =#)).(θlist, Mlist')
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
θfine = range(0, θmax, 40θmax + 1)
sbg(θ) = θ > c4 ? sqrt((1 + θ^2) * (c + θ^2)) / θ : 1 + √(c)
stheory = sbg.(θfine)
bm = s -> "\\mathbf{\\mathit{$s}}"
ylabel = latexstring("\$σ_1($(bm(:Y)))\$ (Avg)")
ps = plot(θfine, θfine, color=:black,
    aspect_ratio = 1, linewidth = 2,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (1,θmax), 1:θmax),
    annotate = (2.1, 3.6, latexstring("c = $c"), :left),
)
plot!(θfine, stheory, color=:blue, label="theory")
scatter!(θlist, σgrid[:,1], marker=:square, color=colors[1], label = labels[1])
scatter!(θlist, σgrid[:,2], marker=:circle, color=colors[2], label = labels[2])

#
prompt()


# u1 plot
ubg(θ) = (θ > c4) ? 1 - c * (1 + θ^2) / (θ^2 * (θ^2 + c)) : 0
utheory = ubg.(θfine)
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
vtheory = vbg.(θfine)
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
    savefig(ps, "gauss1-s.pdf")
    savefig(pu, "gauss1-u.pdf")
    savefig(pv, "gauss1-v.pdf")
end


#=
## Marčenko–Pastur distribution

Examine the singular values of the noise-only matrix ``Z``.
having elements
``z_{ij} ∼ N(0, 1/N)``.
and compare to the asymptotic prediction by the
[Marčenko–Pastur distribution](https://en.wikipedia.org/wiki/Marchenko-Pastur_distribution).
The agreement is remarkably good,
even for a modest matrix size of 100 × 100.
=#

# Marčenko–Pastur pdf
function mp_predict(x::Real, c::Real)
    σm = 1 - sqrt(c)
    σp = 1 + sqrt(c)
    return (σm < x < σp) ?
        sqrt(4c - (x^2 - 1 - c)^2) / (π * c * x) : 0.
end;


function mp_plot(M::Int, N::Int, rando::Function, name::String;
    ntrial = 150,
    bins = range(0, 2, 101),
)
    c = M//N
    pred = mp_predict.(bins, c)
    pmax = ceil(maximum(pred), digits=1)
    data = [svdvals(rando()) for _ in 1:ntrial]
    data = reduce(hcat, data)
    σm = 1 - sqrt(c)
    σp = 1 + sqrt(c)
    xticks = (c == 1) ? (0:2) : round.([0, σm, 1, σp, 2]; digits=2)
    cstr = c == 1 ? L"c = 1" : latexstring("c = $(c.num)/$(c.den) $name")
    histogram(vec(data); bins, linewidth=0,
     xaxis = (L"σ", (0, 2), xticks),
     yaxis = ("", (0, 2.0), [-1, 0, pmax]),
     label = "Empirical", normalize = :pdf,
     left_margin = 10px,
     annotate = (0.1, 1.5, cstr, :left),
    )
    return plot!(bins, pred, label="Predicted")
end;

M = 100
Nlist = [1, 4, 9] * M
fun1 = N -> mp_plot(M, N, () -> randn(M, N) / sqrt(N), "")
pp = fun1.(Nlist)
p3 = plot(pp...; layout=(3,1), size=(600,800))

## savefig(p3, "gauss-mp.pdf")
## savefig(pp[2], "gauss-mp-c4.pdf")

#
prompt()


#=
## Universality

Repeat the previous experiment
with a (zero-mean) Bernoulli distribution.
=#

M = 100
N = 4 * M
randb = () -> rand((-1,1), M, N) / sqrt(N) # Bernoulli, variance 1/N
if false
    tmp = randb()
    @show mean(tmp) # check mean is 0
    @show mean(abs2, tmp), 1/N # check variance is 1/N (exact!)
end
pb = mp_plot(M, N, randb, ", \\mathrm{Bernoulli}")

#
prompt()

# Show a typical Bernoulli matrix realization
pb0 = jim(randb()', "'Bernoulli' matrix"; clim = (-1,1) .* 0.05,
 size=(600,200), right_margin = 30px, cticks=(-1:1)*0.05)


#=
## Sparsity

Universality can break down
if the data is too sparse.
Here we modify Bernoulli to be a categorical distribution
with values
``(-a, 0, a)``
and probabilities
``((1-p)/2, p, (1-p)/2)``,
with ``a`` set so that the variance is ``1/N``.

Here we set ``p`` so that most of the random matrix elements are zero.
In this extremely sparse case,
the Marčenko–Pastur distribution
no longer applies.
=#

M = 100
N = 4 * M
p = (1 - 8/N) # just a few non-zero per row

rands = () -> rand((-1,1), M, N) / sqrt(N * (1-p)) .* (rand(M,N) .> p)
if false
    tmp = rands()
    @show count(==(0), tmp) / (M*N), p
    @show mean(tmp) # check mean is 0
    @show mean(abs2, tmp), 1/N # check variance is 1/N (exact!)
end

# Show a typical matrix realization to illustrate the sparsity
pb1 = jim(rands()', "Very sparse 'Bernoulli' matrix";
 size=(600,200), right_margin = 20px)

# Now make the plot
pss = mp_plot(M, N, rands, ", \\mathrm{Sparse},  p = $p")

#
prompt()

include("../../../inc/reproduce.jl")
