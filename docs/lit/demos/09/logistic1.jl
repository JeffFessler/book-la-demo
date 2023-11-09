#=
# [Logistic regression](@id logistic1)

Binary classification via logistic regression
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
        "Plots"
        "Random"
        "StatsBase"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: dot, eigvals
using MIRTjim: prompt
using Plots: default, gui, savefig
using Plots: plot, plot!, scatter, scatter!
using Random: seed!
using StatsBase: mean
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
    if true # 2017-01-18
        nex = 4 # extra dim (beyond the 2 shown) to make "larger scale"
        v0 = [v0; rand(nex,n0)] # (2+nex, n0)
        v1 = [v1; rand(nex,n1)] # (2+nex, n1)
    end
    M = n0 + n1 # how many samples
    yy = [-ones(n0); ones(n1)] # (M) labels
    vv = [[v0 v1]; ones(1,n0+n1)] # (npar, M) training data
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
    ## savefig(ps, "demo_fgm1_ogm1_s0.pdf")
end
plot(ps)

#
prompt()


#=
# Cost function

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
if !@isdefined(kost)
    pot = (t) -> log(1 + exp(-t)) # logistic
    dpot = (t) -> -1 / (exp(t) + 1)
    tmp = vv * vv' # (npar, npar) covariance
    tmp = eigvals(tmp)
    @show maximum(tmp) / minimum(tmp)
    pLip = maximum(tmp) / 4 # 1/4 comes from logistic curvature

    reg = 2^0
    Lip = pLip + reg # Lipschitz constant

    A = yy .* vv'
    gfun = x -> A' * dpot.(A * x) + reg * x # gradient
    if false
        tmp = gfun(x0)
        @show size(tmp)
    end

    kost = x -> sum(pot, A * x, dims=1) .+ reg/2 * sum(abs2, x, dims=1)
end;


#=
# GD
Iterate GD
=#
tol = 1e-6
tol_gd = tol
tol_n1 = tol
tol_o1 = tol

function gd(x0, gfun::Function, niter::Int)
    xg = copy(x0)
    xgs = copy(xg)
    for n in 1:niter
        xold = xg
        xg -= 1/Lip * gfun(xg)
        if false && (norm(xg - xold, Inf) / norm(x0, Inf) < tol_gd)
            @show n
            break
        end
        any(isnan, xg) && throw("nan")
        ## projected GD ??
        ## xg = xg / norm(xg) # decision boundary unaffected by norm!
        xgs = [xgs xg] # archive
    end
    return xgs
end

if !@isdefined(xgs)
    niter_gd = 300
    xgs = gd(x0, gfun, niter_gd)
    pgs = plot(xgs', xlabel="Iteration", title = "GD")
end
plot(pgs)

#
prompt()

#=
# Nesterov FGM
=#
do_restart = true;

function fgm(x0, grad, Lip::Real, niter::Int)
    re_nest = Int[]
    told = 1
    x = copy(x0)
    zold = copy(x0)
    xns = copy(x)
    for n in 1:niter
        xold = copy(x)
        grad = gfun(x)
        znew = x - 1/Lip * grad
        zdiff = znew - zold # dk - for restart

        tnew = 1/2 * (1 + sqrt(1 + 4 * told^2))

        x = znew + (told - 1) / tnew * (znew - zold)
        zold = copy(znew)
        told = tnew
        if false && (norm(x - xold, Inf) / norm(x0, Inf) < tol_n1)
            @show n
            break
        end
        any(isnan, x) && throw("nan")

        if do_restart && (dot(grad, zdiff) > 0) # dk new version
            told = 1
            x = copy(znew) # check
            zold = copy(x)
            @show "nest. restart", n
            push!(re_nest, n)
        end
        xns = [xns x] # archive
    end
    return xns, re_nest
end

if !@isdefined(xns)
    niter_n1 = 300
    niter_n1 = 500 # nex
    xns, re_nest = fgm(x0, gfun, Lip, niter_n1)
    pns = plot(xns', xlabel = "Iteration", title = "FGM")
end
plot(pns)

#
prompt()


#=
## OGM
Optimized gradient method
=#
function ogm(
    x0,
    gfun,
    Lip::Real,
    niter::Int,
)
    re_ogm1 = Int[]
    told = 1
    x = copy(x0)
    zold = copy(x0)
    x1s = copy(x)
    for n in 1:niter
        xold = x
        grad = gfun(x)
        znew = x - 1/Lip * grad

        tnew = 1/2 * (1 + sqrt(1 + 4 * told^2))

        x = znew + (told - 1) / tnew * (znew - zold) + told/tnew * (znew - xold)
        zdiff = znew - zold # dk - for restart
        zold = znew
        told = tnew
        if false && (norm(x - xold, Inf) / norm(x0, Inf) < tol_n1)
            @show n
            break
        end
        any(isnan, x) && throw("nan")

        if do_restart && (dot(grad, zdiff) > 0) # dk new version
            push!(re_ogm1, n)
            told = 1
            x = znew # check
            zold = x # dk fixed from x0
            @info "ogm1 restart $n"
        end
        x1s = [x1s x] # archive
    end
    return x1s, re_ogm1
end


if !@isdefined(x1s)
    niter_o1 = 300
    niter_o1 = 500 # nex
    x1s, re_ogm1 = ogm(x0, gfun, Lip, niter_o1)
    po1 = plot(x1s', xlabel = "Iteration", title = "OGM")
end
plot(po1)

#
prompt()


# impartial version of x ͚
xh_tmp = [xgs[:,end] xns[:,end] x1s[:,end]]
xh = vec(mean(xh_tmp[:,2:3], dims=2)); # GD too slow to include

# plot cost
extra = do_restart ? " (restart)" : ""
pc = plot(xaxis=("iteration", (0,10)), yaxis=("Cost function",))
plot!(0:niter_gd, vec(kost(xgs)) .- kost(xh), label = "GD" * extra)
plot!(0:niter_n1, vec(kost(xns)) .- kost(xh), label = "FGM" * extra)
plot!(0:niter_o1, vec(kost(x1s)) .- kost(xh), label = "OGM1" * extra)

#
prompt()


# Plot decision boundaries
if true
    psh = deepcopy(ps)
    v2p = @. (-xh[end] - xh[1] * v1p) / xh[2]
    plot!(psh, v1p, v2p, color = :magenta, label="final")
## savefig(psh, "demo-fgm1-fgm1a.pdf")
end
plot(psh)

#
prompt()


#=
# Plot iterate convergence
=#

efun1 = (x) -> vec(sqrt.(sum(abs2, x .- xh, dims=1)))
efun = (x) -> do_restart ? log10.(efun1(x)) : efun1(x)
ifun = (x) -> 0:(size(x,2)-1);

pic = plot(
 xaxis = ("Iteration", (0, 40+10*nex), 0:20:80),
 yaxis = do_restart ?
  (L"\log_{10}(‖ \mathbf{x}_k - \mathbf{x}_* ‖)", (-3, 1), -3:1) :
  (L"‖ \mathbf{x}_k - \mathbf{x}_* ‖", (0, 8), [-1, 0, 8]),
 legend = :topright,
)
plot!(ifun(xgs), efun(xgs), color=:green, label = "GD")
plot!(ifun(xns), efun(xns), color=:blue, label = "Nesterov FGM" * extra)
plot!(ifun(x1s), efun(x1s), color=:red, label = "OGM1" * extra)
if do_restart
    scatter!(re_nest, efun(xns[:, re_nest .+ 1]), color=:blue)
    scatter!(re_ogm1, efun(x1s[:, re_ogm1 .+ 1]), color=:red)
end
plot(pic)

#
prompt()

## savefig demo_fgm1_ogm1c # restart
## savefig demo_fgm1_ogm1b # no restart

## todo: compare with LBFGS

include("../../../inc/reproduce.jl")
