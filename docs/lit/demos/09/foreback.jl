#=
# [Video foreground/background separation](@id foreback)

This example illustrates
video foreground/background separation
via robust PCA
using the Julia language.
For simplicity,
the method here assumes a static camera.
For free-motion camera video, see
[Moore et al., 2019](https://doi.org/10.1109/TCI.2019.2891389).
=#

#srcURL

#=
## Setup
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false # todo
    import Pkg
    Pkg.add([
        "ColorTypes"
        "ColorVectorSpace"
        "Downloads"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "LinearMapsAA"
        "MIRT"
        "MIRTjim"
        "Plots"
        "VideoIO"
    ])
end


# Tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ColorTypes: RGB
using ColorVectorSpace
using Downloads: download
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, I, norm, svd, svdvals
using LinearMapsAA: LinearMapAA, redim
using MIRT: pogm_restart
using MIRTjim: jim, prompt
using Plots: default, gui, plot, savefig
using Plots: gif, @animate, Plots
using VideoIO
default(); default(markerstrokecolor=:auto, label = "")


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Load video data
=#

# Load raw data
if !@isdefined(y1)
    tmp = homedir() * "/111.mp4"
    if !isfile(tmp)
        url = "http://backgroundmodelschallenge.eu/data/synth1/111.mp4"
        tmp = download(url)
    end
    y1 = VideoIO.load(tmp) # 1499 frames of size (480,640)
end;

# convert to arrays
if !@isdefined(Y3)
    tmp = y -> 1f0*permutedims((@view y[1:2:end,1:2:end]), (2,1))
    yf = tmp.(@view y1[1:10:end]) # 150 frames of size (320,240)
    yf = yf[51:end] # 100 frames with moving cars
    Y3 = stack(yf) # (nx,ny,nf)
    (nx, ny, nf) = size(Y3)
end;
    py = jim([yf[1], yf[end], yf[end]-yf[1]];
        nrow = 1, size = (600, 200),
        title="Frame 001  |  Frame $nf  |  Difference")
 

#=
## Cost function
=#

# Encoding operator `A = [I I]` for L+S because we stack `X = [L;;;S]`
tmp = LinearMapAA(I(nx*ny*nf);
    odim=(nx,ny,nf), idim=(nx,ny,nf), T=Float32, prop=(;name="I"))
tmp = kron([1 1], tmp)
A = redim(tmp; odim=(nx,ny,nf), idim=(nx,ny,nf,2)) # "squeeze" odim

unstack(X, i) = selectdim(X, ndims(X), i)
Lpart = X -> unstack(X, 1) # extract "L" from X
Spart = X -> unstack(X, 2) # extract "S" from X
nucnorm(L::AbstractMatrix) = sum(svdvals(L)) # nuclear norm
nucnorm(L::AbstractArray) = nucnorm(reshape(L, :, nf)); # (nx*ny, nf) for L

#=
The robust PCA optimization cost function is:
```math
Ψ(L,S) = \frac{1}{2} ‖ L + S - Y ‖_F^2 + α ‖L‖_* + β ‖vec(S)‖_1
```
or equivalently
```math
Ψ(\mathbf X) =
\frac{1}{2} ‖ [I I] \mathbf{X} - \mathbf{Y} ‖_F^2
 + α ‖ \mathbf{X}[1] ‖_* + β ‖ vec(\mathbf{X}[2]) ‖_1
```

where ``\mathbf Y`` is the original data matrix.
=#

robust_pca_cost(Y, X, α::Real, β::Real) =
    0.5 * norm( A * X - Y )^2 + α * nucnorm(Lpart(X)) + β * norm(Spart(X), 1);


# Proximal algorithm helpers

soft(z,t) = sign(z) * max(abs(z) - t, 0)

# Define singular value soft thresholding (SVST) function
function SVST(X, beta)
    shape = size(X)
    X = reshape(X, :, shape[end]) # unfold
    U,s,V = svd(X)
    sthresh = @. soft(s, beta)
    jj = findall(>(0), sthresh)
    out = U[:,jj] * Diagonal(sthresh[jj]) * V[:,jj]'
    return reshape(out, shape)
end;


#=
## Algorithm
Proximal gradient methods for minimizing
robust PCA cost function
=#
function robust_pca(Y;
    L = Y,
    S = zeros(size(Y)),
    α = 1,
    β = 1,
    mom = :pogm,
    Fcost::Function = X -> robust_pca_cost(Y, X, α, β),
#   fun = (iter, xk, yk, is_restart) -> (),
    fun = mom === :fgm ?
        (iter, xk, yk, is_restart) -> (yk, Fcost(yk), is_restart) :
        (iter, xk, yk, is_restart) -> (xk, Fcost(xk), is_restart),
    kwargs..., # for pogm_restart
)
    
    X0 = stack([L, S])

    f_grad = X -> A' * (A * X - Y) # gradient of smooth term
    f_L = 2 # Lipschitz constant of f_grad
    g_prox = (X, c) -> stack([SVST(Lpart(X), c * α), soft.(Spart(X), c * β)])
    Xhat, out = pogm_restart(X0, Fcost, f_grad, f_L; g_prox, fun, mom, kwargs...)
    return Xhat, out
end

#src # tmp = SVST(Yc, 30)
#src # tmp = Yc .- Yc[:,:,1] # remove background

#=
## Run algorithm
Apply robust PCA to each RGB color channel separately
for simplicity, then reassemble.
=#
if !@isdefined(Xpogm)
    α = 30
    β = 0.1
    niter = 20
    Xc = Array{Any}(undef, 3)
    out = Array{Any}(undef, 3)
    for (i, c) in enumerate([:r :g :b]) # separate colors
        Yc = map(y -> getfield(y, c), Y3);
        Xc[i], out[i] = robust_pca(Yc; α, β, mom = :pogm, niter)
    end
    Xpogm = map(RGB{Float32}, Xc...) # reassemble colors
end

Lpogm = Lpart(Xpogm)
Spogm = Spart(Xpogm)

tmp = cat(dims=1, Y3, Lpogm, Spogm)

#=
## Results
Animate videos
=#
anim1 = @animate for it in 1:nf
    tmp = stack([Y3[:,:,it], Lpogm[:,:,it], Spogm[:,:,it]])
    jim(tmp; nrow=1, title="Original | Low-rank | Sparse",
        xlabel = "Frame $it", size=(600, 250))
#   gui()
end
# gif(anim1; fps = 6)

gui(); throw()
 

#src todo: compare pgm=ista, fpgm=fista, pogm
#src cost_pogm = [o[2] for o in out]


######### old below here

#=
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

#
plot(title = "cost vs. iteration",
    xlabel="iteration", ylabel = "cost function value")
scatter!(cost_ista, label="ISTA", color=:red)
scatter!(cost_fista, label="FISTA", color=:blue)

#
prompt()

# See if the FISTA result is "low rank"
s_fista = svdvals(Xfista)
effective_rank = count(>(0.01*s_fista[1]), s_fista)

#
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

#
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
    pj_pogm = jim(Xpogm, "POGM result at $niter iterations")
end

#
scatter!(pc, 0:niter, cost_pogm, label="POGM", color=:green)

#
prompt()

#src plot(py, pm, pj_ista, pj_fista, pj_admm, pj_pogm)

#todo include("../../../inc/reproduce.jl")
