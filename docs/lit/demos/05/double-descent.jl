#=
# [Double Descent in LS](@id double-descent)

This example illustrates
the phenomenon of double descent
in least squares (LS) polynomial fitting
using the Julia language.

Inspired by the article
["Characterizations of Double Descent"](https://www.siam.org/publications/siam-news/articles/characterizations-of-double-descent)
by Manuchehr Aminian
in SIAM News 58(10) Dec. 2025.

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
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: diag, norm, I, svdvals
using MIRTjim: prompt, jim
using Plots: default, gui, plot, plot!, scatter, scatter!, savefig
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
    markersize = 6, tickfontsize=12, labelfontsize = 16, legendfontsize=14)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);

## Simulate data
M = 100 # number of data points
P = 99 # highest polynomial degree
t = range(-1, 1, M)
#src fun(t) = 2t + cos(25t)
fun(t) = atan(2*t) # nonlinear function
y = fun.(t)
train = 1:(M÷2)
test = (M÷2+1):M;

#=
## Legendre polynomial basis

Build
[Legendre polynomial basis](https://en.wikipedia.org/wiki/Legendre_polynomials)
using
Bonnet's recursion formula:
```math
(n+1) P_{n+1}(x) = (2n+1) x P_n(x) - n P_{n-1}(x)
```
=#
L = ones(M, P)
L[:,2] .= t
for k in 3:P
    n = k - 2 # caution: SIAM article had an error here
    L[:, k] = ((2n+1) * t .* L[:, k-1] - n * L[:, k-2]) / (n + 1)
end
pl = plot(t, L[:,1:5], title="First 5 Legendre polynomials", marker=:dot)

# Check recursion for k=3, corresponding to n=1 in Bonnet's recursion
p2(x) = (1/2) * (3x^2 - 1)
@assert p2.(t) ≈ L[:,3]

# Check basis function normalization (continuous vs discrete)
p = 0:(P-1)
normp = @. sqrt(2 / (2p+1)) # theoretical L₂[-1,1] norm
norme = norm.(eachcol(L)) / sqrt(M/2) # empirical norm, account for dx
plot(xlabel="degree", ylabel="norm")
plot!(p, norme, marker=:dot, color=:red, label="empirical")
plot!(p, normp, marker=:dot, color=:blue, label="analytical")


# Normalize basis functions
# using empirical norms
L = L ./ norme' / sqrt(M/2);

#=
## Scree plot

Examine the singular values of Legendre basis `L`.
Clearly `L` is not semiunitary,
and the last ~15 values are very small. 
So fitting with more than ~80 components
will be very unstable,
even if all `M` samples were available.
=#
scatter(svdvals(L), xaxis = ("k", (0,100), 0:10:100), ylabel = L"σ_k")

#=
Examine orthogonality of the basis functions

The Legendre polynomials are orthogonal in ``L₂[-1,1]``,
but the following correlation figure
shows that they are not orthogonal when sampled.
=#
pc = jim(p, p, L'L, "correlation")


#=
Evaluate OLS solutions for increasing k

The training error decreases monotonically with polynomial degree
=#
errors = zeros(P,3)
for k in 1:P
    A = L[:, 1:k]
    xhat = A[train,:] \ y[train]
    residual = A*xhat - y
    errors[k,1] = norm(residual[train])
    errors[k,2:3] .= norm.((residual[train], residual[test]), Inf)
end
ptrain = scatter(p, 100*errors[:,1]/norm(y[train]), title="NRMSE training",
 xlabel="Polynomial degree")

#=
The test error exhibits double descent
=#
ptest = plot(p, 100*errors[:,3]/norm(y[train]), title="NRMSE test",
 marker=:dot,
 xlabel = "Polynomial degree",
 yaxis = ("NRMSE (%)", (0, 100), ),
)


# Show fits for small, medium and large polynomial degree
pfit = plot(
 xaxis = ("x", (-1,1), -1:1),
 yaxis = ("y", (-1,1) .* 1.5, -1:1),
)
scatter!(t, y)
for k in (2, 10, 99)
   A = L[:, 1:k]
   xhat = A[train,:] \ y[train]
   plot!(t, A * xhat, label="k=$k")
end
pfit

#
include("../../../inc/reproduce.jl")
