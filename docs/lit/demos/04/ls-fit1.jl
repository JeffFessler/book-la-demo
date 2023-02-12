#=
# [LS fitting](@id ls-fit1)

This example illustrates least squares (LS) polynomial fitting
using the Julia language.
=#

#srcURL

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
        "LaTeXStrings"
        "MIRTjim"
        "Random"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: Diagonal, svd
using Random: seed!
using Plots; default(label="", markerstrokecolor=:auto)
#using LaTeXStrings
using MIRTjim: prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Simulated data from latent nonlinear function
s = (t) -> atan(4*(t-0.5)) # nonlinear function

seed!(1) # seed rng
M = 15 # how many data points
tm = sort(rand(M)) # M random sample locations
y = s.(tm) + 0.1 * randn(M) # noisy samples

t0 = LinRange(0, 1, 101) # fine sampling for showing curve
p0 = scatter(tm, y, color=:blue,
    label="y (noisy data)", xlabel="t", ylabel="y", ylim=(-1.3, 1.3))
plot!(t0, s.(t0), color=:blue, label="s(t) : latent signal", legend=:topleft)

#
prompt()


# ## Polynomial fitting

deg = 3 # polynomial degree
Afun = (tt) -> [t.^i for t in tt, i in 0:deg] # matrix of monomials
A = Afun(tm) # M × 4 matrix
p1 = plot(title="Columns of matrix A", xlabel="t", legend=:left)
for i in 0:deg
    plot!(p1, tm, A[:,i+1], marker=:circle, label = "A[:,$(i+1)]")
end

#
prompt()


# ## Fit 4 unknowns with 4 equations

m4 = Int64.(round.(LinRange(1, M-1, 4))) # pick 4 points well separated
A4 = A[m4,:] # 4 × 4 matrix
x4 = inv(A4) * y[m4] # inverse of 4×4 matrix to solve "y = A x"

scatter!(p0, tm[m4], y[m4], marker=:circle, color=:red)
plot!(p0, t0, Afun(t0)*x4, color=:red, label="Fit using 4 of $M points")

#
prompt()


# ## Fit 4 unknowns using all M=15 equations

xh = A \ y # backslash for LS solution using all M samples

plot!(p0, t0, Afun(t0)*xh, color=:green, label="Fit cubic using all M=$(M) points")

#
prompt()

#src savefig("tmp.pdf")



# ## SVD solution

U, s, V = svd(A)
s


# ## Verify equivalence of SVD and backslash solutions to LS problem

xh2 = V * Diagonal(1 ./ s) * (U' * y) # SVD-based solution
xh3 = V * ( (1 ./ s) .* (U' * y) ) # mathematically equivalent alternate expression

@assert xh ≈ xh2
@assert xh ≈ xh3


include("../../../inc/reproduce.jl")
