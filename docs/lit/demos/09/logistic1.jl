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
#       "StatsBase"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings # nice plot labels
using LinearAlgebra: dot, eigvals
using MIRTjim: jim, prompt
#todo using MLDatasets: MNIST
using Plots: default, gui, savefig
using Plots: plot, plot!, scatter, scatter!
#using Plots: RGB, cgrad
#using Plots.PlotMeasures: px
using Random: seed! #, randperm
using StatsBase: mean
default(); default(markersize=6, linewidth=2, markerstrokecolor=:auto, label="",
 tickfontsize=12, labelfontsize=18, legendfontsize=18, titlefontsize=18)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

#todo isinteractive() ? jim(:prompt, true) : prompt(:draw);


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
ps
prompt()

#=
# Cost function
todo: equation
Lipschitz constant
=#
# Lipschitz constant
if !@isdefined(kost)
    pot = (t) -> log(1 + exp(-t)) # logistic
    dpot = (t) -> -1 / (exp(t) + 1)
    tmp = vv * vv' # (npar, npar) covariance
    tmp = eigvals(tmp)
    @show maximum(tmp) / minimum(tmp)
    pLip = maximum(tmp) / 4 # 1/4 comes from logistic curvature

    reg = 2^0
    Lip = pLip + reg

    A = yy .* vv'
    gfun = x -> A' * dpot.(A * x) + reg * x
    tmp = gfun(x0)
    @show size(tmp)

#   kost = x -> pot.(x' * vv * yy)' + reg/2 * sum(abs2, x, dims=1)
    kost = x -> ones(M)' * pot.(A * x) .+ reg/2 * sum(abs2, x, dims=1)
# todo    kost = x -> sum(pot, A * x, dims=1) .+ reg/2 * sum(abs2, x, dims=1)
end

#=
# GD
Iterate GD
=#

tol = 1e-6
tol_gd = tol
tol_n1 = tol
tol_o1 = tol
#todo   do_restart = false
do_restart = true

if !@isdefined(xgs)
    niter_gd = 300
    xg = copy(x0)
    xgs = copy(xg)
    for n in 1:niter_gd
        xold = xg
        global xg -= 1/Lip * gfun(xg)
        if false && (norm(xg - xold, Inf) / norm(x0, Inf) < tol_gd)
            @show n
            break
        end
        if any(isnan, xg)
            throw("nan")
        end
        # projected GD ??
#       xg = xg / norm(xg); # decision boundary not affected by norm!?
        global xgs = [xgs xg] # archive
    end
end

if false
    v2p = @. (-xg[end] - xg[1] * v1p) / xg[2]
    psg = deepcopy(ps)
    plot!(psg, v1p, v2p, color = :magenta, label="final")

    plot(xgs', title = "GD") # check convergence
end


#=
# Nesterov FGM
=#

if !@isdefined(xns)
    re_nest = Int[]
    niter_n1 = 300
    niter_n1 = 500 # nex
    told = 1
    x = copy(x0)
    zold = copy(x0)
    xns = copy(x)
    for n in 1:niter_n1
        xold = copy(x)
        grad = gfun(x)
        znew = x - 1/Lip * grad
        zdiff = znew - zold # dk - for restart

        tnew = 1/2 * (1 + sqrt(1 + 4 * told^2))

        global x = znew + (told - 1) / tnew * (znew - zold)
        global zold = copy(znew)
        global told = tnew
        if false && (norm(x - xold, Inf) / norm(x0, Inf) < tol_n1)
            @show n
            break
        end
        if any(isnan, x)
            throw("nan")
        end

    #    if do_restart && (dot(grad, x - xold) > 0) # jf old version
    #    if do_restart && (dot(grad, x - znew) > 0) # dk new version
        if do_restart && (dot(grad, zdiff) > 0) # dk new version
            told = 1
            x = copy(znew) # check
            zold = copy(x)
            @show "nest. restart", n
            push!(re_nest, n)
        end
        global xns = [xns x] # archive
    end
    xn = xns[:,end]
end

## plot(xns', title = "FGM")

#=
## OGM
todo
=#
function ogm(
    x0,
    gfun,
    Lip::Real,
    ;
#   niter_o1 = 300,
    niter_o1 = 500, # nex
)
    re_ogm1 = Int[]
    told = 1
    x = copy(x0)
    zold = copy(x0)
    x1s = copy(x)
    for n in 1:niter_o1
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

    ##   if do_restart && (dot(grad, x - xold) > 0) # jf old version
    ##   if do_restart && (dot(grad, x - znew) > 0) # dk new version
        if do_restart && (dot(grad, zdiff) > 0) # dk new version
            push!(re_ogm1, n)
            told = 1
            x = znew # check
            zold = x # dk fixed from x0
            @info("ogm1 restart $n")
        end
        x1s = [x1s x] # archive
    end
    return x1s, re_ogm1
end

if !@isdefined(x1s)
    x1s, re_ogm1 = ogm(x0, gfun, Lip)
    ## plot(x1s')
end

prompt()

xh_tmp = [xgs[:,end] xns[:,end] x1s[:,end]]
xh = vec(mean(xh_tmp[:,2:3], dims=2)) # GD too slow to include

# plot cost
plot(xaxis=("iteration", (0,10)), )
plot!(0:niter_gd, vec(kost(xgs)) .- kost(xh), label="GD")
plot!(0:niter_n1, vec(kost(xns)) .- kost(xh), label="FGM")
plot!(0:niter_o1, vec(kost(x1s)) .- kost(xh), label="OGM1")


# Plot decision boundaries
if true
    psn = deepcopy(ps)
    v2p = @. (-xh[end] - xh[1] * v1p) / xh[2]
    plot!(psn, v1p, v2p, color = :magenta, label="final")
## savefig(psn, "demo-fgm1-fgm1a.pdf")
end
prompt()


#=
# plot iterate convergence
=#

plot(
 xaxis = ("Iteration", (0, 40+10*nex), 0:20:80),
#src yaxis = ("‖ \\mathbf{x}_k - \\mathbf{x}_* ‖", (0, 8), [-1, 0, 8]),
 yaxis = (L"\log_{10}(‖ x_k - x_* ‖)", (-3, 1), -3:1),
#legend = :bottomleft,
)

efun = (x) -> log10.(vec(sqrt.(sum(abs2, x .- xh, dims=1))))
ifun = (x) -> 0:(size(x,2)-1)

plot!(ifun(xgs), efun(xgs), color=:green, label="GD")
plot!(ifun(xns), efun(xns), color=:blue, label="Nesterov FGM (restart)")
plot!(ifun(x1s), efun(x1s), color=:red, label="OGM1 (restart)")
if do_restart
    scatter!(re_nest, efun(xns[:, re_nest .+ 1]), color=:blue)
    scatter!(re_ogm1, efun(x1s[:, re_ogm1 .+ 1]), color=:red)
end

#   if ~do_restart
#       ytick([0 8])
#   end
#   if do_restart
    #    ir_savefig cw demo_fgm1_ogm1c # restart
#   else
#       legend('GD', 'Nesterov FGM', 'OGM1') # no restart
    #    ir_savefig cw demo_fgm1_ogm1b # no restart
#   end

#
prompt()

#todo include("../../../inc/reproduce.jl")
