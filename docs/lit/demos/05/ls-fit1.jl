#=
# [LS fitting](@id ls-fit1)

This example illustrates least squares (LS) polynomial fitting
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
using LinearAlgebra: Diagonal, svd
using MIRTjim: prompt
using Plots: default, plot, plot!, scatter, scatter!, savefig
using Random: seed!
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
    markersize = 6, tickfontsize=12, labelfontsize = 16, legendfontsize=14)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Simulated data from latent nonlinear function
s = (t) -> atan(4*(t-0.5)) # nonlinear function

seed!(0) # seed rng
M = 15 # how many data points
tm = sort(rand(M)) # M random sample locations
y = s.(tm) + 0.1 * randn(M) # noisy samples

t0 = range(0, 1, 101) # fine sampling for showing curve
xaxis=(L"t", (0,1), 0:0.5:1)
yaxis=(L"y", (-1.3, 1.3), -1:1)
p0 = scatter(tm, y, color=:black, label="y (noisy data)"; xaxis, yaxis)
plot!(t0, s.(t0), color=:blue, label="s(t) : latent signal", legend=:topleft)

#
prompt()


# ## Polynomial fitting

deg = 3 # polynomial degree
Afun = (tt) -> [t.^i for t in tt, i in 0:deg] # matrix of monomials
A = Afun(tm) # M × 4 matrix
p1 = plot(title="Columns of matrix A", xlabel=L"t", legend=:left)
for i in 0:deg
    plot!(p1, tm, A[:,i+1], marker=:circle, label = "A[:,$(i+1)]")
end
p1

#
prompt()


# ## Fit 4 unknowns with 4 equations

m4 = Int64.(round.(range(1, M-1, 4))) # pick 4 points well separated
A4 = A[m4,:] # 4 × 4 matrix
x4 = inv(A4) * y[m4] # inverse of 4×4 matrix to solve "y = A x"

p1 = scatter(tm[m4], y[m4], marker=:square, color=:red)
scatter!(tm, y, color=:black, label="y (noisy data)"; xaxis, yaxis)
plot!(t0, s.(t0), color=:blue, label="s(t) : latent signal", legend=:topleft)
plot!(t0, Afun(t0)*x4, color=:red, label="Fit 4 of $M points")

#
prompt()


# ## Fit 4 unknowns using all M=15 equations

xh = A \ y # backslash for LS solution using all M samples

plot!(p1, t0, Afun(t0)*xh, color=:magenta, label="Fit cubic to M=$(M) points")

#
prompt()

#src savefig(p1, "ls-fit-4-15.pdf")



# ## SVD solution

U, s, V = svd(A)
s


# ## Verify equivalence of SVD and backslash solutions to LS problem

xh2 = V * Diagonal(1 ./ s) * (U' * y) # SVD-based solution
xh3 = V * ( (1 ./ s) .* (U' * y) ) # mathematically equivalent alternate expression

@assert xh ≈ xh2
@assert xh ≈ xh3


include("../../../inc/reproduce.jl")
