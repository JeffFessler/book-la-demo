#=
# [Random matrix theory and rank-1 signal + noise](@id gauss1)

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
using LinearAlgebra: dot, rank, svd
using MIRTjim: prompt
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

c = 1 # square matrices for simplicity
c4 = c^0.25

function rmt1(T::Type{<:Real}, n::Int, θ::Real) # one trial
    m = n # c = 1
    u = randn(T, m) / T(sqrt(m))
    v = randn(T, n) / T(sqrt(n))
    X = θ * u * v' # theoretically rank-1 matrix
    Z = randn(T, m, n) / T(sqrt(n))
    Y = X + Z
    fac = svd(Y)
    σ1 = fac.S[1]
    u1 = fac.U[:,1]
    v1 = fac.Vt[1,:]
    return [σ1, abs2(dot(u1, u)), abs2(dot(v1, v))]
end;

function rmt2(T::Type{<:Real}, n::Int, θ::Real, nrep::Int) # avg nrep trials
    sum = [0, 0, 0]
    for _ in 1:nrep
        sum += rmt1(T, n, θ)
    end
    return sum / nrep
end;

# Simulation parameters
T = Float32
nlist = [30, 300]
nn = length(nlist)
θmax = 3
nθ = θmax * 4 + 1
nrep = 100
θlist = T.(range(0, θmax, nθ));

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

include("../../../inc/reproduce.jl")
