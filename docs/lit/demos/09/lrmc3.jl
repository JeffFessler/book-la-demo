#=
# [Low-rank matrix completion](@id lrmc3)

This example illustrates
low-rank matrix completion
using the Julia language.

History:
- 2017-11-07 Greg Ongie, University of Michigan, original version
- 2017-11-12 Jeff Fessler, minor modifications
- 2021-08-23 Julia 1.6.2
- 2023-04-11 Literate version for Julia 1.8

=#

#srcURL

#=
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "DelimitedFiles"
        "Downloads"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRT"
        "MIRTjim"
        "Plots"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using DelimitedFiles: readdlm
using Downloads: download
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: svd, svdvals, Diagonal, norm
using MIRT: pogm_restart
using MIRTjim: jim, prompt
using Plots: plot, scatter, scatter!, savefig, default
default(markerstrokecolor=:auto, label = "")


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);



#=
## `TOP SECRET`

On 2017-11-06 10:34:12 GMT,
Agent 556 obtained a photo of the illegal arms dealer,
code name **Professor X-Ray**.
However, Agent 556 spilled soup on their lapel-camera,
shorting out several CCD sensors.
The image matrix has several missing entries;
we were only able to recover 25% of data!

Agent 551: Your mission, should you choose to accept it,
is to recover the missing entries and uncover the true identity
of **Professor X-Ray**.

Read in data with missing pixels set to zero
=#

if !@isdefined(Y)
    tmp = homedir() * "/web/course/551/julia/demo/professorxray.txt" # jf
    if !isfile(tmp)
        url = "https://web.eecs.umich.edu/~fessler/course/551/julia/demo/professorxray.txt"
        tmp = download(url)
    end
    Y = collect(readdlm(tmp)')
    py = jim(Y, "Y: Corrupted image matrix of Professor X-Ray\n (missing pixels set to 0)")
end


# Create binary mask ``Ω`` (true=observed, false=unobserved)
Ω = Y .!= 0
percent_nonzero = sum(Ω) / length(Ω) # count proportion of missing entries

# Show mask
pm = jim(Ω, "Ω: Locations of observed entries")


#=
## Low-rank approximation
A simple low-rank approximation works poorly
for this much missing data
=#
r = 20
U,s,V = svd(Y)
Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]'
pr = jim(Xr, "Low-rank approximation for r=$r")


#=
## Low-rank matrix completion

Instead, we will try to uncover the identity of
**Professor X-Ray** using
low-rank matrix completion.

The optimization problem we will solve is:
```math
\min_{\mathbf X}
\frac{1}{2} ‖ P_Ω(\mathbf X) - P_Ω(\mathbf Y) ‖_2^2
+ β ‖ \mathbf X ‖_*
\quad\quad\text{(NN-min)}
```
where ``\mathbf Y`` is the zero-filled input data matrix,
and ``P_Ω`` is the operator
that extracts a vector of entries belonging to the index set ``Ω``.

Define cost function for optimization problem:
=#

nucnorm = (X) -> sum(svdvals(X))
costfun = (X, beta) -> 0.5 * norm(X[Ω] - Y[Ω])^2 + beta * nucnorm(X);

# Define singular value soft thresholding (SVST) function
function SVST(X, beta)
    U,s,V = svd(X)
    sthresh = @. max(s - beta, 0)
    return U * Diagonal(sthresh) * V'
end;


#=
## Iterative Soft-Thresholding Algorithm (ISTA)

ISTA is an extension of gradient descent to convex cost functions
that look like
``\min_x f(x) + g(x)``
where ``f(x)`` is smooth and ``g(x)`` is non-smooth.
Also known as a
[proximal gradient method](https://en.wikipedia.org/wiki/Proximal_gradient_methods_for_learning).

**ISTA algorithm for solving (NN-min):**
- initialize ``\mathbf X_0 = \mathbf Y`` (zero-fill missing entries)
- `for` ``k=0,1,2,…``
  - ``[\hat{\mathbf X}_k]_{i,j} = \begin{cases}[\mathbf X_k]_{i,j} & (i,j) ∉ Ω \\ [\mathbf Y]_{i,j} & (i,j) ∈ Ω \end{cases}``
    (Put back in known entries)

  - ``\mathbf X_{k+1} = \text{SVST}(\hat{\mathbf X}_k,β)``
    (Singular value soft-thresholding)
- `end`

Apply ISTA:
=#
niter = 400
beta = 0.01 # chosen by trial-and-error here
function lrmc_ista(Y)
    X = copy(Y)
    Xold = copy(X)
    cost_ista = zeros(niter+1)
    cost_ista[1] = costfun(X,beta)
    for k in 1:niter
        X[Ω] = Y[Ω]
        X = SVST(X,beta)
        cost_ista[k+1] = costfun(X,beta)
    end
    return X, cost_ista
end;

if !@isdefined(Xista)
    Xista, cost_ista = lrmc_ista(Y)
    pj_ista = jim(Xista, "ISTA result at $niter iterations")
end


#=
What went wrong? Let's investigate.
First, let's see if the above solution is actually low-rank.
=#

s_ista = svdvals(Xista)
s0 = svdvals(Y)
plot(title = "singular values",
    xtick = [1, sum(s .> 20*eps()), minimum(size(Y))])
scatter!(s0, color=:black, label="Y (initialization)")
scatter!(s_ista, color=:red, label="X (ISTA)")

#
prompt()

# Now let's check the cost function descent:
scatter(cost_ista, color=:red,
    title = "cost vs. iteration",
    xlabel = "iteration",
    ylabel = "cost function value",
    label = "ISTA",
)

#
prompt()

#=
## Fast Iterative Soft-Thresholding Algorithm (FISTA)

Modification of ISTA
that includes Nesterov acceleration for faster convergence.

Reference:
- Beck, A. and Teboulle, M., 2009.
  [A fast iterative shrinkage-thresholding algorithm for linear inverse problems.](https://doi.org/10.1137/080716542)
  SIAM J. on Imaging Sciences, 2(1), pp.183-202.

**FISTA algorithm for solving (NN-min)**
- initialize matrices ``\mathbf Z_0 = \mathbf X_0 = \mathbf Y``
- `for` ``k=0,1,2,…``
  - ``[\hat{\mathbf Z}_k]_{i,j} = \begin{cases}[\mathbf Z_k]_{i,j} & (i,j) ∉ Ω \\ [\mathbf Y]_{i,j} & (i,j) ∈ Ω \end{cases}``
    (Put back in known entries)

  - ``\mathbf X_{k+1} = \text{SVST}(\hat{\mathbf Z}_k,\beta)``

  - ``t_{k+1} = \frac{1 + \sqrt{1+4t_k^2}}{2}`` (Nesterov step-size)

  - ``\mathbf Z_{k+1} = \mathbf X_{k+1} + \frac{t_k-1}{t_{k+1}}(\mathbf X_{k+1}-\mathbf X_{k})`` (Momentum update)
- `end`

Run FISTA:
=#

niter = 200
function lrmc_fista(Y)
    X = copy(Y)
    Z = copy(X)
    Xold = copy(X)
    told = 1
    cost_fista = zeros(niter+1)
    cost_fista[1] = costfun(X,beta)
    for k in 1:niter
        Z[Ω] = Y[Ω]
        X = SVST(Z,beta)
        t = (1 + sqrt(1+4*told^2))/2
        Z = X + ((told-1)/t)*(X-Xold)
        Xold = X
        told = t
        cost_fista[k+1] = costfun(X,beta) # comment out to speed-up
    end
    return X, cost_fista
end;

if !@isdefined(Xfista)
    Xfista, cost_fista = lrmc_fista(Y)
    pj_fista = jim(Xfista, "FISTA result at $niter iterations")
end

plot(title = "cost vs. iteration",
    xlabel="iteration", ylabel = "cost function value")
scatter!(cost_ista, label="ISTA", color=:red)
scatter!(cost_fista, label="FISTA", color=:blue)

#
prompt()

# See if the FISTA result is "low rank"
s_fista = svdvals(Xfista)
effective_rank = count(>(0.01*s_fista[1]), s_fista)

ps = plot(title="singular values",
    xtick = [1, effective_rank, count(>(20*eps()), s_fista), minimum(size(Y))])
scatter!(s0, label="Y (initial)", color=:black)
scatter!(s_fista, label="X (FISTA)", color=:blue)

#
prompt()

# Exercise: think about why ``σ_1(X) > σ_1(Y)`` !


#=
## Alternating directions method of multipliers (ADMM)

ADMM is another approach that uses SVST as a sub-routine,
closely related to proximal gradient descent.

It is faster than FISTA,
but the algorithm requires a tuning parameter ``μ``.
(Here we use ``μ = β``).

References:
- Cai, J.F., Candès, E.J. and Shen, Z., 2010.
  [A singular value thresholding algorithm for matrix completion.](https://doi.org/10.1137/080738970)
  SIAM J. Optimization, 20(4), pp. 1956-1982.
- Boyd, S., Parikh, N., Chu, E., Peleato, B. and Eckstein, J., 2011.
  [Distributed optimization and statistical learning via the alternating direction method of multipliers.](https://doi.org/10.1561/2200000016)
  Foundations and Trends in Machine Learning, 3(1), pp. 1-122.

Run alternating directions method of multipliers (ADMM) algorithm:
=#

niter = 50
# Choice of parameter ``μ`` can greatly affect convergence rate
function lrmc_admm(Y; mu::Real = beta)
    X = copy(Y)
    Z = zeros(size(X))
    L = zeros(size(X))
    cost_admm = zeros(niter+1)
    cost_admm[1] = costfun(X,beta)
    for k in 1:niter
        Z = SVST(X + L, beta / mu)
        X = (Y + mu * (Z - L)) ./ (mu .+ Ω)
        L = L + X - Z
        cost_admm[k+1] = costfun(X,beta) # comment out to speed-up
    end
    return X, cost_admm
end;

if !@isdefined(Xadmm)
    Xadmm, cost_admm = lrmc_admm(Y)
    pj_admm = jim(Xadmm, "ADMM result at $niter iterations")
end

pc = plot(title = "cost vs. iteration",
    xtick = [0, 50, 200, 400],
    xlabel = "iteration", ylabel = "cost function value")
scatter!(0:400, cost_ista, label="ISTA", color=:red)
scatter!(0:200, cost_fista, label="FISTA", color=:blue)
scatter!(0:niter, cost_admm, label="ADMM", color=:magenta)

#
prompt()

# All singular values
s_admm = svdvals(Xadmm)
scatter!(ps, s_admm, label="X (ADMM)", color=:magenta, marker=:square)

#
prompt()

#=
For a suitable choice of ``μ``, ADMM converges faster than FISTA.
=#


#=
## Proximal optimized gradient method (POGM)

The
[proximal optimized gradient method (POGM)](https://doi.org/10.1137/16m108104x)
with
[adaptive restart](https://doi.org/10.1007/s10957-018-1287-4)
is faster than FISTA
with very similar computation per iteration.
Unlike ADMM,
POGM does not require any algorithm tuning parameter ``μ``,
making it easier to use in many practical composite optimization problems.
=#

if !@isdefined(Xpogm)
    Fcost = X -> costfun(X, beta)
    f_grad = X -> Ω .* (X - Y) # gradient of smooth term
    f_L = 1 # Lipschitz constant of f_grad
    g_prox = (X, c) -> SVST(X, c * beta)
    fun = (iter, xk, yk, is_restart) -> (xk, Fcost(xk), is_restart)
    niter = 150
    Xpogm, out = pogm_restart(Y, Fcost, f_grad, f_L; g_prox, fun, niter)
    cost_pogm = [o[2] for o in out]
end
pj_pogm = jim(Xpogm, "POGM result at $niter iterations")

scatter!(pc, 0:niter, cost_pogm, label="POGM", color=:green)

#
prompt()

#src plot(py, pm, pj_ista, pj_fista, pj_admm, pj_pogm)

include("../../../inc/reproduce.jl")
