#=
# [Rank-1 approximation](@id rank-1)

This example illustrates rank-1 approximations
using the Julia language.
=#

#srcURL

#=
## Setup
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
    ])
end


# Tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: svd, rank
using MIRTjim: prompt
using Plots: default, plot!, scatter, scatter!, savefig
using Random: seed!
default(); default(label="", markerstrokecolor=:auto,
    guidefontsize=14, legendfontsize=14, tickfontsize=12)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Generate data

# Noisy data with slope=1.  Both x and y values are noisy!
seed!(0)
x0 = 1:8 # true x locations
x = x0 + 2*randn(size(x0)) # called "errors in variables"
y = x0 + 2*randn(size(x0)); # noisy samples

# Plotting utility function
lineplot = (p, s, c, l; w=3, t=:dash) ->
    plot!(p, 0:10, (0:10)*s, line=(c,t), label=l, width=w)
function plotdata()
    p = scatter(x, y, label="data", legend=:bottomright,
        color=:blue, markersize=7, aspect_ratio=:equal,
        xaxis = (L"x", (0, 10), 0:4:8),
        yaxis = (L"y", (0, 10), 0:4:8),
    )
    lineplot(p, 1, :red, "true", t=:solid, w=2)
end
pl = plotdata()

#
prompt()


# ## Rank-1 approximation

# To make a low-rank approximation, collect data into a matrix
A = [x'; y']

# Examine singular values
U, s, V = svd(A)
s # 2nd singular value is much smaller than 1st


# Construct rank-1 approximation
B = U[:,1] * s[1] * V[:,1]' # rank-1 approximation
rank(B)

#
B


# ### Plot rank-1 approximation
xb = B[1,:]
yb = B[2,:]

lineplot(pl, (xb\yb)[1], :black, "")
scatter!(pl, xb, yb, color=:black, markersize=5, marker=:square, label="rank1")

#
prompt()


# ## Use least-squares estimation to estimate slope:
slope = y'*x / (x'*x) # cf inv(A'A) * A'b
slope = (x \ y)[1] # cf A \ b


# ### Plot the LS fit and the low-rank approximation on same graph
pa = lineplot(pl, slope, :green, "LS")

#
prompt()

## savefig(pa, "06_low_rank1_all.pdf")


# ## Illustrate the Frobenius norm approximation error graphically
pf = plotdata()
for i in 1:length(xb)
    plot!(pf, [x[i], xb[i]], [y[i], yb[i]], color=:black, width=2)
end
lineplot(pf, (xb\yb)[1], :black, "")
scatter!(pf, xb, yb, color=:black, markersize=5, marker=:square, label="rank1")

#
prompt()

## savefig(pf, "06_low_rank1_r1.pdf")


# ## Illustrate the LS residual graphically
xl = x; yl = slope*xl # LS points
ps = plotdata()
for i in 1:length(x)
    plot!(ps, [x[i], xl[i]], [y[i], yl[i]], color=:green, width=2)
end
lineplot(ps, slope, :green, "")
scatter!(ps, xl, yl, color=:green, markersize=5, marker=:square, label="LS")

#
prompt()

## savefig(ps, "06_low_rank1_ls.pdf")


include("../../../inc/reproduce.jl")
