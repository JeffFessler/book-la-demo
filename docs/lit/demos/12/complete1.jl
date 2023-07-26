#=
# [RMT and matrix completion](@id complete1)

This example examines matrix completion
(recovery of a low-rank matrix from data with missing measurements)
with the lens of random matrix theory,
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

#todo isinteractive() && prompt(:prompt);



#=
## Helper functions
=#

c = 1 # square matrices for simplicity
c4 = c^0.25

function gen1( ; # random data for one trial
    T::Type{<:Real} = Float32,
    θ::Real = 3,
    p_obs::Real = 0.8, # probability an element is observed
    N::Int = 100,
    M::Int = N,
)
    c = M/N
    mask = rand(M, N) .< p_obs
    u = rand((-1,+1), M) / T(sqrt(M))
    v = rand((-1,+1), N) / T(sqrt(N))
#   u = randn(T, m) / T(sqrt(m))
#   v = randn(T, n) / T(sqrt(n))
    X = θ * u * v' # theoretically rank-1 matrix
    Y = mask .* X # missing entries set to zero
    return Y, u, v, θ
end;

function trial1( ; kwargs...) # svd results for 1 trial
    Y, u, v, θ = gen1( ; kwargs...)
    fac = svd(Y)
    σ1 = fac.S[1]
    u1 = fac.U[:,1]
    v1 = fac.Vt[1,:]
    return [σ1, abs2(dot(u1, u)), abs2(dot(v1, v))]
end

function trial2( # avg nrep trials
    nrep::Int;
    T::Type{<:Real} = Float32,
    θ::Real = 3,
    N::Int = 100,
    kwargs...,
)
    sum = [0, 0, 0]
    for _ in 1:nrep
        sum += trial1( ; T, θ, N, kwargs...)
    end
    return sum / nrep
end;

# SVD for each of multiple trials, for different SNRs and matrix sizes
function trial3(
    Nlist::Vector{<:Int},

    labels = Vector{LaTeXString}(undef, nn)
    σgrid = zeros(nθ, nn)
    ugrid = zeros(nθ, nn)
    vgrid = zeros(nθ, nn)
    for (i_n, n) in enumerate(nlist) # sizes
        labels[i_n] = latexstring("\$N = $n\$")
        for (it, θ) in enumerate(θlist) # SNRs
            ## @show i_n, it
            tmp = rmt2(T, n, θ, nrep)
            σgrid[it, i_n] = tmp[1]
            ugrid[it, i_n] = tmp[2]
            vgrid[it, i_n] = tmp[3]
        end
    end
end

# Simulation parameters
T = Float32
p_obs = 0.8
Nlist = [30, 300]
nn = length(Nlist)
θmax = 3
nθ = θmax * 4 + 1
nrep = 100
θlist = T.(range(0, θmax, nθ));

N = 30
tmp = trial2(nrep; T, N, θ, p_obs)

throw()
# jim(Y)

# SVD for each of multiple trials, for different SNRs and matrix sizes
if !@isdefined(ugrid)
    labels = Vector{LaTeXString}(undef, nn)
    σgrid = zeros(nθ, nn)
    ugrid = zeros(nθ, nn)
    vgrid = zeros(nθ, nn)
    for (i_n, n) in enumerate(nlist) # sizes
        labels[i_n] = latexstring("\$N = $n\$")
        for (it, θ) in enumerate(θlist) # SNRs
            ## @show i_n, it
            tmp = rmt2(T, n, θ, nrep)
            σgrid[it, i_n] = tmp[1]
            ugrid[it, i_n] = tmp[2]
            vgrid[it, i_n] = tmp[3]
        end
    end
end


#=
## σ1 plot
Compare theory and empirical results.
=#
colors = [:orange, :red]
θfine = range(0, θmax, 50θmax + 1)
sbg(θ) = θ > c4 ? sqrt((1 + θ^2) * (c + θ^2)) / θ : 1 + √(c)
stheory = sbg.(θfine)
ylabel = latexstring("\$σ_1(Y)\$ (Avg)") # of $nrep trials)")
ps = plot(θlist, θlist, color=:black, linewidth=2, aspect_ratio = 1,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (1,θmax), 1:θmax),
)
plot!(θfine, stheory, color=:blue, label="theory")
scatter!([θlist[1]], [σgrid[1,1]], marker=:square, color=colors[1],
 label = labels[1])
scatter!([θlist[1]], [σgrid[1,2]], marker=:circle, color=colors[2],
 label = labels[2])
plot!(θlist, σgrid[:,1], marker=:square, color=colors[1])
plot!(θlist, σgrid[:,2], marker=:circle, color=colors[2])

#
prompt()


# u1 plot
ubg(θ) = (θ > c4) ? 1 - c * (1 + θ^2) / (θ^2 * (θ^2 + c)) : 0
utheory = ubg.(θfine)
ylabel = latexstring("\$|⟨\\hat{u}, u⟩|^2\$ (Avg)")# of $nrep trials)")
pu = plot(θfine, utheory, color=:blue, label="theory", left_margin = 10px,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (0,1), 0:0.5:1),
)
scatter!([θlist[1]], [ugrid[1,1]], marker=:square, color=colors[1],
 label = labels[1])
scatter!([θlist[1]], [ugrid[1,2]], marker=:circle, color=colors[2],
 label = labels[2])
plot!(θlist, ugrid[:,1], marker=:square, color=colors[1])
plot!(θlist, ugrid[:,2], marker=:circle, color=colors[2])

#
prompt()


# v1 plot
pow = 1.0
vbg(θ) = ( (θ > c^0.25) ? 1 - (c + θ^2) / (θ^2 * (θ^2 + 1)) : 0 )^pow
vtheory = @. vbg(θfine)^pow
vgr = vgrid.^pow
ylabel = latexstring("\$|⟨\\hat{v}, v⟩|^2\$ (Avg)")# of $nrep trials)")
pv = plot(θfine, vtheory, color=:blue, label="theory", left_margin = 10px,
    xaxis = (L"θ", (0,3), 0:3),
    yaxis = (ylabel, (0,1), 0:0.5:1),
)
scatter!([θlist[1]], [vgr[1,1]], marker=:square, color=colors[1],
 label = labels[1])
scatter!(θlist, vgr[:,2], marker=:circle, color=colors[2],
 label = labels[2])
plot!(θlist, vgr[:,1], marker=:square, color=colors[1])
plot!(θlist, vgr[:,2], marker=:circle, color=colors[2])

#
prompt()


## savefig(ps, "gauss1-s.pdf")
## savefig(pu, "gauss1-u.pdf")
## savefig(pv, "gauss1-v.pdf")


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
pj = jim(rands()', "Very sparse 'Bernoulli' matrix";
 size=(600,200), right_margin = 20px)

# Now make the plot
ps = mp_plot(M, N, rands, ", \\mathrm{Sparse},  p = $p")

#
prompt()

include("../../../inc/reproduce.jl")
