#=
# [Markov chain](@id markovchain)

This example illustrates a Markov chain
using the Julia language.

In this demo,
the elements of the transition matrix ``P`` are
``p_{ij} = p(X_{k+1} = i | X_{k} = j)``.
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
        "Random"
        "Statistics"
        "StatsBase"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: eigen
using MIRTjim: prompt
using Plots: annotate!, default, plot, plot!, scatter, scatter!
using Plots: gif, @animate, Plots
using Random: seed!
using StatsBase: wsample

default(
 markerstrokecolor = :auto, label="",
 labelfontsize=8, legendfontsize=8, size = (1,1) .* 600,
)
seed!(0);


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# Transition matrix
p = 0.4
P = [
 0 0.0 1 0;
 1 0.0 0 0;
 0 1-p 0 1;
 0  p  0 0;
];


# Define plot helpers
color = [:red, :green, :blue, :purple];

function plot_circle!(x, y, ic=0; r=10)
    t = range(0, 2π, 101)
    plot!(x .+ r * cos.(t), y .+ r * sin.(t), color=color[ic], width=2)
    annotate!(x, y+0.7r, ("$ic", 14, color[ic]))
end;

xc = [-1, 1, -1, 1] * 20
yc = [1, 1, -1, -1] * 20

# function for plotting the Markov chain diagram
function plot_chain(steps::Int)

    plot(xtick=nothing, ytick=nothing, axis=:off, aspect_ratio=1,)
    plot_circle!.(xc, yc, 1:4)

    for ii in 1:4
        if P[ii,ii] != 0
            @show "Bug"
            #add_loop!()
        end
        for jj in 1:4
            ((ii == jj) || P[ii,jj] == 0) && continue
            xi = xc[ii]
            xj = xc[jj]
            yi = yc[ii]
            yj = yc[jj]
            xd = xi - xj
            yd = yi - yj
            phi = atan(yd, xd) + π/2
            rd = sqrt(xd^2 + yd^2)
            frac = (rd - 1*10) / rd
            xs = frac * xj + (1-frac) * xi
            xe = frac * xi + (1-frac) * xj
            ys = frac * yj + (1-frac) * yi
            ye = frac * yi + (1-frac) * yj
            xm = (xi + xj) / 2
            ym = (yi + yj) / 2
            xo = 5 * cos(phi)
            yo = 5 * sin(phi)
            plot!([xs, xe], [ys, ye], arrow=:arrow, color=:black, width=2)
            annotate!(xm+xo, ym+yo, ("$(P[ii,jj])", 13))
        end
    end
    title = steps > 0 ? "$steps steps" : "Initial state"
    plot!(; title)
end;



#=
### Initial conditions
This block starts the simulation
with an initial grid of 100 particles
=#
xg = repeat(-4.5:4.5, 1, 10)
yg = xg'
xg = xg[:]
yg = yg[:]
node = fill(2, length(xg)) # all particles start in node (state) X_0 = 2
plot_chain(0)
scatter!(xc[node] + xg[:], yc[node] + yg[:])

#
#src prompt()

# Use `wsample` (weighted sampling) for transitions
function node_update!(node)
    for kk in 1:length(node)
        node[kk] = wsample(1:size(P,1), P[:,node[kk]]) # random process
    end
end;

function run_and_plot(iter::Int)
    node_update!(node)
    plot_chain(iter)
    scatter!(xc[node] + xg[:], yc[node] + yg[:])
    for ii in 1:4
        tmp = sum(node .== ii) / length(node)
        annotate!(xc[ii]+10, yc[ii]-10, ("$tmp", 7, color[ii]))
    end
    plot!()
end;


# Simulate
anim1 = @animate for iter in [1:20; 30:10:100]
    run_and_plot(iter)
end
gif(anim1; fps = 4)



#=
### Clicker question 1
The transition matrix P in this example is (choose most specific correct answer):
- A. Square
- B. Nonnegative
- C. Irreducible
- D. Primitive
- E. Positive"
=#

#=
### Clicker question 2 (later)
Which state has the lowest probability in equilibrium?  
- A 1  
- B 2  
- C 3  
- D 4  
- E None: they are all equally likely"
=#

#=
# function to nicely print the eigenvector matrix
function matprint(V)
    for i in 1:size(V,1)
        for j in 1:size(V,2)
            v = V[i,j]
            @printf("%5.2f", real(v))
            print(imag(v) < 0 ? " -" : " +")
            @printf("%5.2fı  ", abs(imag(v)))
        end
        println()
    end
end;
=#

# Eigenvectors:
(d, V) = eigen(P)
round.(V; digits=3)

# Eigenvalues:
[d abs.(d)] # exactly one λ=1 and only one |λ| = 1

# Plot eigenvalues in complex plane
scatter(real(d), imag(d), color=:blue,
 xaxis = ("Re(λ)", -1:1),
 yaxis = ("Im(λ)", -1:1),
 framestyle = :origin,
 size = (400,400),
)
tmp = range(0, 2π, 301)
plot!(cos.(tmp), sin.(tmp), color=:black)

#
prompt()


# Steady-state distribution 
v = real(V[:,4])
πss = v / sum(v) # normalize
[πss; "check:"; 1 / (p + 4 - 1); p / (p + 4 - 1)]

#=
For insight:
# 4^2 - 2*4 + 2 # N^2 - 2N + 2 in Ch8
=#
P^10

# Approximate limiting distribution
P^200


include("../../../inc/reproduce.jl")
