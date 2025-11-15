#=
# [Low-rank matrix completion: AltMin, ISTA, FISTA](@id lrmc-m)

This example illustrates
low-rank matrix completion
via alternating projection, ISTA (PGM), and FISTA (FPGM),
using the Julia language.

(This approach is related to "projection onto convex sets" (POCS) methods,
but the term "POCS" would be a misnomer here
because the rank constraint is not a convex set.)

History:
* 2021-08-23 Julia 1.6.2
* 2021-12-09 Julia 1.6.4 and use M not Ω
* 2023-06-04 Julia 1.9.0 in Literate

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
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LinearAlgebra: svd, svdvals, rank, norm, Diagonal
using LaTeXStrings
using MIRTjim: jim, prompt
using Plots: default, gui, plot, savefig, scatter, scatter!, xlabel!, xticks!
using Plots: RGB, cgrad
using Plots.PlotMeasures: px
using Random: seed!
using Statistics: mean
default(
 markersize=7, markerstrokecolor=:auto, label = "",
 tickfontsize = 10, legendfontsize = 18, labelfontsize = 16, titlefontsize = 18,
)

# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);
jim(:prompt, true);


#=
## Latent matrix
Make a matrix that has low rank
=#
tmp = [
    zeros(1,20);
    0 1 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 1 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 1 0 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0;
    0 0 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0;
    zeros(1,20)
]';
rank(tmp)

#
Xtrue = kron(10 .+ 80*tmp, ones(9,9))
rtrue = rank(Xtrue)

# Helper functions for
# plots with consistent size
jim1 = (X ; kwargs...) -> jim(X; size = (600,300),
 leftmargin = 10px, rightmargin = 10px, kwargs...);
# and consistent display range
jimc = (X ; kwargs...) -> jim1(X; clim=(0,100), kwargs...);
# and with NRMSE label
nrmse = (Xh) -> round(norm(Xh - Xtrue) / norm(Xtrue) * 100, digits=1)
args = (xaxis = false, yaxis = false, colorbar = :none) # book
args = (;) # web
jime = (X; kwargs...) -> jimc(X; xlabel = "NRMSE = $(nrmse(X)) %",
 args..., kwargs...,
)
title = latexstring("\$\\mathbf{\\mathit{X}}\$ : Latent image")
pt = jimc(Xtrue; title, xlabel = " ", args...)
## savefig(pt, "mc_ap_x.pdf")


#=
## Noisy / incomplete data
=#
seed!(0)
M = rand(Float32, size(Xtrue)) .>= 0.75 # 75% missing
Y = M .* (Xtrue + randn(size(Xtrue)));

title = latexstring("\$\\mathbf{\\mathit{Y}}\$ : Corrupted image matrix\n(missing pixels set to 0)")
py = jime(Y ; title)
## savefig(py, "mc_ap_y.pdf")

#=
### What is rank(Y) ??
* A 5-9
* B 10-49
* C 50-59
* D 60-70
* E 71-200

`rank(Y)`
`svdvals(Y)`
=#

# Show mask, count proportion of missing entries
frac_nonzero = count(M) / length(M)
title = latexstring("\$\\mathbf{\\mathit{M}}\$ : Locations of observed entries")
pm = jim1(M; title, args...,
    xlabel = "sampled fraction = $(round(frac_nonzero * 100, digits=1))%")
## savefig(pm, "mc_ap_m.pdf")

#=
## Low-rank approximation
A simple low-rank approximation works poorly for missing data.
=#
r = 5
U,s,V = svd(Y)
Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]'
title = latexstring("Rank $r approximation of data \$\\mathbf{\\mathit{Y}}\$")
pr = jime(Xr ; title)
## savefig(pr, "mc_ap_lr.pdf")


#=
## Alternating projection
Alternating projection is an
iterative method that alternates
between projecting onto the set 𝒞 of rank-5 matrices
and onto the set 𝒟 of matrices that match the data.
=#

function projC(X, r::Int)
    U,s,V = svd(X)
    return U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]' # project onto "𝒞" &Cscr; U+1D49E
end;

function lrmc_alt(Y, r::Int, niter::Int)
    Xr = copy(Y)
    Xr[.!M] .= mean(Y[M]) # init: fill missing values with mean of other values
    @show nrmse(Xr)
    for iter in 1:niter
        Xr = projC(Xr, r) # project onto "𝒞" &Cscr; U+1D49E
        Xr[M] .= Y[M] # project onto "𝒟" &Dscr; U+1D49F
        if 0 == iter % 40
            @show nrmse(Xr)
        end
    end
    return Xr
end;

niter_alt = 400
r = 5
Xr = lrmc_alt(Y, r, niter_alt)
title = "Alternating Projection at $niter_alt iterations"
pa = jime(Xr ; title)
## savefig(pa, "mc_ap_400.pdf")


#=
### What is rank(Xr) here ??
* A 5-9
* B 10-49
* C 50-59
* D 60-70
* E 71-200

`rank(Xr)`
`svdvals(Xr)`
=#

# Run one more projection step onto the set of rank-r matrices
Xfinal = projC(Xr, r)
pf = jime(Xfinal ; title="Alternating Projection at $niter_alt iterations")
## savefig(pf, "mc_ap_xh.pdf")

#=
### What is rank(Xfinal) here ??
* A 5-9
* B 10-49
* C 50-59
* D 60-70
* E 71-200

`rank(Xfinal)`
=#


# Plot singular values
sr = svdvals(Xr)
rankeff = s -> count(>(0.01*s[1]), s); # effective rank

#
ps = plot(title="singular values",
 xaxis=(L"k", (1, minimum(size(Y))), [1, rankeff(sr), minimum(size(Y))]),
 yaxis=(L"σ",), labelfontsize = 18,
 leftmargin = 15px, bottommargin = 20px, size = (600,350), widen = true,
)
scatter!(ps, svdvals(Y), color=:red, label="Y (data)", marker=:dtriangle)
scatter!(ps, svdvals(Xtrue), color=:blue, label="Xtrue", marker=:utriangle)
pa = deepcopy(ps)
scatter!(pa, sr, color=:green, label="Alt. Proj. output")

## savefig(pa, "mc_ap_sv.pdf")

# ## Think about why ``σ₁(Y) ≪ σ₁(X_{\mathrm{true}})``
prompt()

#=
## Nuclear norm approach
Now we will try to recover the matrix
using low-rank matrix completion
with a nuclear-norm regularizer.

The optimization problem we will solve is:
```math
\arg\min_{\mathbf{\mathit{X}}} \frac{1}{2}
‖ \mathbf{\mathit{M}} ⊙
 (\mathbf{\mathit{X}} - \mathbf{\mathit{Y}}) ‖_{\mathrm{F}}^2
+ \beta ‖ \mathbf{\mathit{X}} ‖_*
\quad\quad (\text{NN-min})
```
* ``\mathbf{\mathit{Y}}``
  is the zero-filled input data matrix
* ``\mathbf{\mathit{M}}``
  is the binary sampling mask.

Define cost function for optimization problem
=#
nucnorm = (X) -> sum(svdvals(X)) # nuclear norm
costfun1 = (X,beta) -> 0.5 * norm(M .* (X - Y))^2 + beta * nucnorm(X); # regularized cost


#=
### Q. The cost function above is (convex, strictly convex):
* A: F,F
* B: F,T
* C: T,F
* D: T,T
=#

# Define singular value soft thresholding (SVST) function
function SVST(X::AbstractMatrix, beta::Real)
    U,s,V = svd(X) # see below
    sthresh = @. max(s - beta, 0)
    index = findall(>(0), sthresh)
    return (@view U[:,index]) * Diagonal(sthresh[index]) * (@view V[:,index])'
end;

#=
### Q. Which SVD is that?
* A compact
* B economy
* C full
* D SUV
* E none of these

- `U,s,V = svd(Y)`
- `@show size(s), size(U), size(V)`
=#


#=
## ISTA

The iterative soft-thresholding algorithm (ISTA)
is an extension of gradient descent
for (often convex) "composite" cost functions that look like
``\min_x f(x) + g(x)``
where ``f(x)`` is smooth and ``g(x)`` is non-smooth.

ISTA is also known as the
[proximal gradient method (PGM)](http://www.stat.cmu.edu/~ryantibs/convexopt-S15/lectures/08-prox-grad.pdf).

ISTA algorithm for solving (NN-min):
* Initialize ``\mathbf{\mathit{X}}_0 = \mathbf{\mathit{Y}}`` (zero-fill missing entries)
* `for k=0,1,2,...`
* ``[\mathbf{\mathit{X}}_k]_{i,j} =
  \begin{cases}[\mathbf{\mathit{X}}_k]_{i,j} & \text{if } (i,j) ∉ Ω
  \\ [\mathbf{\mathit{Y}}]_{i,j} & \text{if } (i,j) ∈ Ω \end{cases}``
  (Put back in known entries)

* ``\mathbf{\mathit{X}}_{k+1} = \text{SVST}(\mathbf{\mathit{X}}_k, \beta)``
  (Singular value soft-thresholding)
* `end`
=#

# ISTA for matrix completion, using functions `SVST` and `costfun1`
function lrmc_ista(Y, M, beta::Real, niter::Int)
    X = copy(Y)
    Xold = copy(X)
    cost = zeros(niter+1)
    cost[1] = costfun1(X, beta)
    for k in 1:niter
        @. X[M] = Y[M] # in place
        X = SVST(X, beta)
        cost[k+1] = costfun1(X, beta)
    end
    return X, cost
end;

# Apply ISTA
niter = 1000
beta = 0.8 # chosen by trial-and-error here
xh_ista, cost_ista = lrmc_ista(Y, M, beta, niter)
pp = jime(xh_ista ; title="ISTA result at $niter iterations")

## savefig(pp, "mc-nuc-ista.pdf")


#=
That result is not good.
What went wrong? Let's investigate.
First, check if the ISTA solution is actually low-rank.
=#

sp = svdvals(xh_ista)
psi = deepcopy(ps)
scatter!(psi, sp, color=:orange, label=L"\hat{X} \mathrm{(ISTA)}")
xticks!(psi, [1, rtrue, rank(Diagonal(sp)), minimum(size(Y))])

#
prompt()

# Now check the cost function.
# It is decreasing monotonically, but quite slowly.

scatter(cost_ista, color=:orange,
    title="cost vs. iteration",
    xlabel="iteration",
    ylabel="cost function value",
    label="ISTA")

#
prompt()


#=
## FISTA

The fast iterative soft-thresholding algorithm (FISTA)
is a modification of ISTA that includes Nesterov acceleration
for much faster convergence.
Also known as the fast proximal gradient method (FPGM).

Reference:
- Beck, A. and Teboulle, M., 2009.
  [A fast iterative shrinkage-thresholding algorithm for linear inverse problems](https://doi.org/10.1137/080716542).
  SIAM journal on imaging sciences, 2(1):183-202.

**FISTA algorithm for solving (NN-min)**
* initialize matrices
  ``\mathbf Z_0 = \mathbf X_0 = \mathbf Y``
* `for k=0,1,2,...`
* ``[\mathbf{Z}_k]_{i,j} =
  \begin{cases}[\mathbf Z_k]_{i,j} & \text{if}~(i,j) ∉ Ω
  \\ [\mathbf{Y}]_{i,j} & \text{if}~(i,j) ∈ Ω \end{cases}``
  (Put back in known entries)

* ``\mathbf{X}_{k+1} = \text{SVST}(\mathbf{Z}_k, \beta)``
* ``t_{k+1} = \frac{1 + \sqrt{1+4t_k^2}}{2}`` (Nesterov step-size)
* ``\mathbf{Z}_{k+1} = \mathbf{X}_{k+1} + \frac{t_k-1}{t_{k+1}} (\mathbf{X}_{k+1} - \mathbf{X}_k)``
  (Momentum update)
* `end`
=#

# FISTA algorithm for low-rank matrix completion, using `SVST` and `costfun1`
function lrmc_fista(Y, M, beta::Real, niter::Int)
    X = copy(Y)
    Z = copy(X)
    Xold = copy(X)
    told = 1
    cost = zeros(niter+1)
    cost[1] = costfun1(X, beta)
    for k in 1:niter
        @. Z[M] = Y[M]
        X = SVST(Z, beta)
        t = (1 + sqrt(1+4*told^2))/2
        Z = X + ((told-1)/t) * (X - Xold)
        Xold = copy(X)
        told = t
        cost[k+1] = costfun1(X, beta) # comment out to speed-up
    end
    return X, cost
end;

# Run FISTA
niter = 300
xh_nn_fista, cost_fista = lrmc_fista(Y, M, beta, niter)
p1 = jime(xh_nn_fista ; title="FISTA with nuclear norm at $niter iterations")

## savefig(p1, "lrmc-nn-fs300.pdf")


#=
Plot showing that FISTA converges much faster!
[POGM](https://doi.org/10.1007/s10957-018-1287-4)
would be even faster.
=#
plot(title="cost vs. iteration for NN regularizer",
    xlabel="iteration", ylabel="cost function value")
scatter!(cost_ista, color=:orange, label="ISTA")
scatter!(cost_fista, color=:magenta, label="FISTA")

#
prompt()

# See if the FISTA result is "low rank"
sf = svdvals(xh_nn_fista)
rfista = rank(Diagonal(sf))
rfista, rankeff(sf)

#
psf = deepcopy(ps)
scatter!(psf, sf, color=:magenta, label="Xh (output of FISTA)")
xticks!(psf, [1, rtrue, rfista, minimum(size(Y))])

#
prompt()

#=
* Optional exercise: think about why ``σ_1(Y) ≪ σ_1(\hat{X}) < σ_1(X_{\mathrm{true}})``
* Optional: try ADMM too
=#


#=
### Your work goes below here
The results below are place-holders that will be much improved
when implemented properly.
=#

if true # replace these place-holder functions with your work
    shrink_p_1_2(v, reg::Real) = v
    lr_schatten(Y, reg::Real) = Y
    fista_schatten(Y, M, reg::Real, niter::Int) = Y
else # instructor version
    mydir = ENV["hw551test"] # change path
    include(mydir * "shrink_p_1_2.jl") # 1D shrinker for |x|^(1/2), previous HW
    include(mydir * "lr_schatten.jl")
    include(mydir * "fista_schatten.jl")
end;

# Apply FISTA for Schatten p=1/2
niter = 100
reg_fs = 120
xh_fs = fista_schatten(Y, M, reg_fs, niter)

p2 = jime(xh_fs; title="FISTA for Schatten p=1/2, $niter iterations")
## savefig("schatten_complete_fs150_sp.pdf")


# See if the Schatten FISTA result is "low rank"
ss = svdvals(xh_fs)
rank_schatten_fista = rank(Diagonal(ss))
rank_schatten_fista, rankeff(ss)

pss = deepcopy(ps)
scatter!(pss, ss, color=:cyan, label="Xh (FISTA for Schatten)")
xticks!(pss, [1, rank_schatten_fista, minimum(size(Y))])

#
prompt()

# red-black-blue colormap
RGB255(args...) = RGB((args ./ 255)...)
color = cgrad([RGB255(230, 80, 65), :black, RGB255(23, 120, 232)])

# Error image for nuclear norm
p3 = jimc(xh_nn_fista - Xtrue; title = "FISTA Nuclear Norm: Xh-X",
 clim=(-80,80), color)
## savefig(p3, "schatten_complete_fs300_nn_err.pdf")

# Error image for schatten p=1/2
p4 = jimc(xh_fs - Xtrue; title = "FISTA Schatten p=1/2 'Norm': Xh-X",
 clim=(-80,80), color)
## savefig(p4, "schatten_complete_fs150_sp_err.pdf")

# Cost function plot
if false # set to true for HW
   costfun2 = (X,β) -> 0.5 * norm(M .* (X - Y))^2 + β * norm(svdvals(X), 1/2)
   tmp = niter -> costfun2( fista_schatten(Y, M, reg_fs, niter), reg_fs )
   niter2 = 100
   cost_fista2 = tmp.(0:niter)
   p5 = scatter(0:niter, cost_fista2,
    title="cost vs. iteration for FISTA Schatten",
    xlabel="iteration",
    ylabel="cost function value",
    label="FISTA Schatten",
   )
## savefig(p5, "schatten_complete_fs100_cost.pdf")
end

include("../../../inc/reproduce.jl")
