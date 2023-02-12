#=
# [Rank-1 approximation](@id rank-1)

This example illustrates rank-1 approximations
using the Julia language.

This entire page was generated using a single Julia file:
[rank1.jl](@__REPO_ROOT_URL__/06/rank1.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`rank1.ipynb`](@__NBVIEWER_ROOT_URL__/06/rank1.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`rank1.ipynb`](@__BINDER_ROOT_URL__/06/rank1.ipynb),

#=
First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "LinearAlgebra"
        "Plots"
        "Random"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: svd, rank
using Random: seed!
using Plots; default(label="", markerstrokecolor=:auto)
#src using LaTeXStrings
using MIRTjim: prompt
using InteractiveUtils: versioninfo


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
        color=:blue, markersize=7,
        aspect_ratio=:equal, xtick=0:4:8, ytick=0:4:8,
        xlabel="x", ylabel="y", xlim=(0,10), ylim=(0,10))
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
lineplot(pl, slope, :green, "LS")

#
prompt()

#src savefig("06_low_rank1_all.pdf")


# ## Illustrate the Frobenius norm approximation error graphically
pl = plotdata()
for i in 1:length(xb)
    plot!(pl, [x[i], xb[i]], [y[i], yb[i]], color=:black, label="", width=2)
end
lineplot(pl, (xb\yb)[1], :black, "")
scatter!(pl, xb, yb, color=:black, markersize=5, marker=:square, label="rank1")

#
prompt()

#src savefig("06_low_rank1_r1.pdf")


# ## Illustrate the LS residual graphically
xl = x; yl = slope*xl # LS points
pl = plotdata()
for i in 1:length(x)
    plot!(pl, [x[i], xl[i]], [y[i], yl[i]], color=:green, label="", width=2)
end
lineplot(pl, slope, :green, "")
scatter!(pl, xl, yl, color=:green, markersize=5, marker=:square, label="LS")

#
prompt()

#src savefig("06_low_rank1_ls.pdf")


include("../../../inc/reproduce.jl")
