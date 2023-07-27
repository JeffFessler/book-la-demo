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
##  u = randn(T, M) / T(sqrt(M))
##  v = randn(T, N) / T(sqrt(N))
    X = θ * u * v' # theoretically rank-1 matrix
    Z = randn(M, N) / sqrt(N) # gaussian noise
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
end

# Average `nrep` trials:
trial2(nrep::Int, args...) = mean((_) -> trial1(args...), 1:nrep);


# SVD for each of multiple trials, for different SNRs and matrix sizes:
if !@isdefined(vgrid)

    ## Simulation parameters
    T = Float32
    p_obs = 0.64
    Mlist = [30, 300]
    θmax = 4
    nθ = θmax * 4 + 1
    nrep = 100
    θlist = T.(range(0, θmax, nθ));
    labels = map(n -> latexstring("\$M = $n\$"), Mlist)

    c = 0.5 # non-square matrix to test
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
θfine = range(0, 2θmax, 50θmax + 1)
θmod = θfine .* sqrt(p_obs) # key modification from RMT!
sbg(θ) = θ > c4 ? sqrt((1 + θ^2) * (c + θ^2)) / θ : 1 + √(c)
stheory = sbg.(θmod) * sqrt(p_obs) # note modification!
bm = s -> "\\mathbf{\\mathit{$s}}"
ylabel = latexstring("\$σ_1($(bm(:Y)))\$ (Avg)")
ps = plot(θfine, stheory, color=:blue, label="theory",
    aspect_ratio = 1, legend = :topleft,
    xaxis = (L"θ", (0,θmax), 0:θmax),
    yaxis = (ylabel, (1,θmax), 1:θmax),
    annotate = (4.1, 4.5, latexstring("c = $c"), :left),
)
scatter!(θlist, σgrid[:,1], marker=:square, color=colors[1], label = labels[1])
scatter!(θlist, σgrid[:,2], marker=:circle, color=colors[2], label = labels[2])
tmp = latexstring("θ p, p = $p_obs")
plot!(θlist, θlist * p_obs, label=tmp, color=:black, linewidth=2)

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


include("../../../inc/reproduce.jl")
