#=
# [Logistic regression - QN](@id logistic2)

Binary classification via logistic regression
using Quasi-Newton optimizer
in Julia.
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
        "Optim"
        "Plots"
        "Random"
        "Statistics"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: dot, eigvals
using MIRTjim: prompt
using Optim: optimize
import Optim # Options
using Plots: default, gui, savefig
using Plots: histogram!, plot, plot!, scatter, scatter!
using Random: seed!
using Statistics: mean
default(); default(markersize=6, linewidth=2, markerstrokecolor=:auto, label="",
 tickfontsize=12, labelfontsize=18, legendfontsize=18, titlefontsize=18)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? prompt(:prompt) : prompt(:draw)


#=
## Data
Generate synthetic data from two classes
=#
if !@isdefined(yy)
    seed!(0)
    n0 = 60
    n1 = 50
    mu0 = [-1, 1]
    mu1 = [1, -1]
    v0 = mu0 .+ randn(2,n0) # class -1
    v1 = mu1 .+ randn(2,n1) # class 1
    nex = 0
    if false
        nex = 4 # extra dim (beyond the 2 shown) to make "larger scale"
        v0 = [v0; rand(nex,n0)] # (2+nex, n0)
        v1 = [v1; rand(nex,n1)] # (2+nex, n1)
    end
    M = n0 + n1 # how many samples
    yy = [-ones(Int, n0); ones(Int, n1)] # (M) labels
    vv = [[v0 v1]; ones(1,M)] # (npar, M) training data - with bias/offset
    npar = 3 + nex # unknown parameters
end;


# scatter plot and initial decision boundary
if !@isdefined(ps)
    x0 = [-1; 3; rand(nex); 5]
    v1p = range(-1,1,101) * 4
    v2p_fun = x -> @. (-x[end] - x[1] * v1p) / x[2]

    ps = plot(aspect_ratio = 1, size = (550, 500), legend=:topright,
     xaxis = (L"v_1", (-4, 4), [-4 -1 0 1 4]),
     yaxis = (L"v_2", (-4, 4), [-4 -1 0 1 4]),
    )
    plot!(v1p, v2p_fun(x0), color=:red, label="initial")
    plot!(v1p, v1p, color=:yellow, label="ideal")
    scatter!(v0[1,:], v0[2,:], color=:green, alpha=0.7)
    scatter!(v1[1,:], v1[2,:], color=:blue, marker=:square, alpha=0.7)
end
plot(ps)

#
prompt()


#=
## Cost function

Logistic regression with Tikhonov regularization:
```math
f(x) = 1_M' h.(A x) + β/2 ‖ x ‖_2^2
```
where
``h(z) = log(1 + e^{-z})``
is the logistic loss function.

Its gradient is
``∇ f(x) = A' \dot{h}.(A x) + β x``,
and its Lipschitz constant
is ``‖A‖_2^2 / 4 + β``.
=#
if !@isdefined(cost)
    pot(t) = log(1 + exp(-t)) # logistic
    dpot(t) = -1 / (exp(t) + 1)
    tmp = vv * vv' # (npar, npar) covariance
    tmp = eigvals(tmp)
    @show maximum(tmp) / minimum(tmp)
    pLip = maximum(tmp) / 4 # 1/4 comes from logistic curvature

    reg = 0 # no regularization because N ≪ M here
    Lip = pLip + reg # Lipschitz constant

    A = yy .* vv'
    gfun = x -> A' * dpot.(A * x) + reg * x # gradient
    if false
        tmp = gfun(x0)
        @show size(tmp)
    end

    cost(x::AbstractVector) = sum(pot, A * x) + reg/2 * sum(abs2, x)
    cost(x::AbstractMatrix) = cost.(eachcol(x)) ## to handle arrays
end;


#=
## L-BFGS optimizer
=#
opt = Optim.Options(
 store_trace = true,
 show_warnings = false,
 extended_trace = true, # for trace of x
)
outq = optimize(cost, gfun, x0, opt; inplace=false)
xqs = hcat(Optim.x_trace(outq)...)
xq = outq.minimizer


xh = xqs[:,end] # final estimate

# Plot cost
ifun = xs -> 0:(size(xs,2)-1)
pc = plot(xaxis=("iteration", (0,16), 0:4:16), yaxis=("Cost function",))
plot!(ifun(xqs), cost(xqs) .- cost(xh), label = "QN", marker=:o)

#
prompt()


# Plot decision boundaries
if true
    psh = deepcopy(ps)
    v2p = @. (-xh[end] - xh[1] * v1p) / xh[2]
    plot!(psh, v1p, v2p, color = :magenta, label="final")
end
plot(psh)

#
prompt()


#=
## Plot iterate convergence
=#

efun1 = (x) -> vec(sqrt.(sum(abs2, x .- xh, dims=1)))
efun = (x) -> log10.(efun1(x))
pic = plot(
xaxis = ("Iteration", (0, 16), 0:2:16),
 yaxis = (L"\log_{10}(‖ \mathbf{x}_k - \mathbf{x}_* ‖)", (-9, 3), -9:3),
 legend = :topright,
)
plot!(ifun(xqs), efun(xqs), label = "QN", marker = :o)
plot(pic)

#
prompt()


#=
## Plot 1D separation
=#

inprod0 = [v0; ones(1,n0)]' * xh
inprod1 = [v1; ones(1,n1)]' * xh

accuracy0 = round(count(<(0), inprod0) / n0 * 100, digits=1)
accuracy1 = round(count(>(0), inprod1) / n1 * 100, digits=1)

plot(xaxis=("⟨x,v⟩",))
bins = -15:15
histogram!(inprod0, alpha=0.5; bins, color=:green, linecolor = :green,
 label="class 0: $accuracy0%")
histogram!(inprod1, alpha=0.5; bins, color=:blue, linecolor = :blue,
 label="class 1: $accuracy1%")

#
prompt()


#=
## Method
Stand-alone function for (regularized) logistic regression
=#

"""
   xh = logistic(data, label, reg)

Perform regularized logistic regression for binary `label`s
by minimizing
``f(x) = 1_M' h.(A x) + β/2 ‖ x ‖_2^2``
where
``h(z) = log(1 + e^{-z})``
is the logistic loss function.

In:
- `data` `N × M` where `N` is number of features (including offset)
- `label` vector of `M` labels ±1
- `reg` regularization parameter

Out:
- `xh` minimizer of ``f``
"""
function logistic(data::AbstractMatrix, labels::AbstractVector, reg::Real)
    any(x -> ∉(x, (-1,1)), labels) && throw("labels must be ±1")
    pot(t) = log(1 + exp(-t)) # logistic
    dpot(t) = -1 / (exp(t) + 1) # derivative
    tmp = data * data' # (N, N) covariance
    tmp = eigvals(tmp)
    pLip = maximum(tmp) / 4 # 1/4 comes from logistic curvature
    Lip = pLip + reg # Lipschitz constant

    A = labels .* data'
    cost(x) = sum(pot, A * x) + reg/2 * sum(abs2, x)
    gfun(x) = A' * dpot.(A * x) + reg * x # gradient

    x0 = zeros(size(data,1))
    outq = optimize(cost, gfun, x0; inplace=false)
    return outq.minimizer
end;

xl = logistic(vv, yy, reg)
@assert xl ≈ xh

include("../../../inc/reproduce.jl")
