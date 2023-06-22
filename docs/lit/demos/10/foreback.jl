#=
# [Video foreground/background separation](@id foreback)

This example illustrates
video foreground/background separation
via
[robust PCA](https://en.wikipedia.org/wiki/Robust_principal_component_analysis)
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

if false
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
        "Statistics"
        "VideoIO"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ColorTypes: RGB, N0f8
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
using Statistics: mean
using VideoIO
default(); default(markerstrokecolor=:auto, label = "", markersize=6,
legendfontsize = 8) # todo: https://github.com/JuliaPlots/Plots.jl/issues/4621


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Load video data
=#

# Load raw data
if !@isdefined(y1)
    url = "https://github.com/JeffFessler/book-mmaj-data/raw/main/data/bmc-12/111-240-320-100.mp4"
    tmp = download(url)
    y1 = VideoIO.load(tmp) # 100 frames of size (240,320)
    if !isinteractive() # downsample for github cloud
        y1 = map(y -> y[1:2:end,1:2:end], (@view y1[2:2:end]))
    end
end;

# Convert to array
yf = map(y -> 1f0*permutedims(y, (2,1)), y1) # 100 frames of size (320,240)
Y3 = stack(yf) # (nx,ny,nf)
(nx, ny, nf) = size(Y3)
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
Ψ(\mathbf{L},\mathbf{S}) = \frac{1}{2}
 \| \mathbf{L} + \mathbf{S} - \mathbf{Y} \|_{\mathrm{F}}^2
 + α \| \mathbf{L} \|_* + β \| \mathrm{vec}(\mathbf{S}) \|_1
```
or equivalently
```math
Ψ(\mathbf{X}) = \frac{1}{2}
 \left\| [\mathbf{I} \ \mathbf{I}] \mathbf{X} - \mathbf{Y} \right\|_{\mathrm{F}}^2
 + α \| \mathbf{X}_1 \|_* + β \| \mathrm{vec}(\mathbf{X}_2) \|_1
```
where ``\mathbf{Y}`` is the original data matrix.
=#

robust_pca_cost(Y, X, α::Real, β::Real) =
    0.5 * norm( A * X - Y )^2 + α * nucnorm(Lpart(X)) + β * norm(Spart(X), 1);


# Proximal algorithm helpers:
soft(z,t) = sign(z) * max(abs(z) - t, 0);

# Singular value soft thresholding (SVST) function:
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
robust PCA cost function.

The
[proximal optimized gradient method (POGM)](https://doi.org/10.1137/16m108104x)
with
[adaptive restart](https://doi.org/10.1007/s10957-018-1287-4)
is faster than FISTA
with very similar computation per iteration.
POGM does not require any algorithm tuning parameter,
making it easier to use than ADMM
in many practical composite optimization problems.
=#
function robust_pca(Y;
    L = Y,
    S = zeros(size(Y)),
    α = 1,
    β = 1,
    mom = :pogm,
    Fcost::Function = X -> robust_pca_cost(Y, X, α, β),
##  fun = (iter, xk, yk, is_restart) -> (),
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
end;

#src # tmp = SVST(Yc, 30)
#src # tmp = Yc .- Yc[:,:,1] # remove background

#=
## Run algorithm
Apply robust PCA to each RGB color channel separately
for simplicity, then reassemble.
=#
channels = [:r :g :b]
if !@isdefined(Xpogm)
    α = 30
    β = 0.1
    niter = 10
    Xc = Array{Any}(undef, 3)
    out = Array{Any}(undef, 3)
    for (i, c) in enumerate(channels) # separate color channels
        @info "channel $c"
        Yc = map(y -> getfield(y, c), Y3);
        Xc[i], out[i] = robust_pca(Yc; α, β, mom = :pogm, niter)
    end
    Xpogm = map(RGB{Float32}, Xc...) # reassemble colors
end;

#=
## Results
=#

# Extract low-rank (background) and sparse (foreground) components:
Lpogm = Lpart(Xpogm)
Spogm = Spart(Xpogm)
iz = nf * 81 ÷ 100
tmp = stack([Y3[:,:,iz], Lpogm[:,:,iz], Spogm[:,:,iz]])
jim(:line3type, :white)
pf = jim(tmp; nrow=1, size=(700, 250),
  title="Original frame $iz | low-rank background | sparse foreground")
## savefig(pf, "foreback-81.pdf")


# Cost function plot
# overall cost for all 3 color channels
tmp = sum(out -> [o[2] for o in out], out)
pc = plot(0:niter, tmp;
  xlabel="Iteration", ylabel="Cost function", marker = :circle, label="POGM")

#
prompt()


#=
## Alternatives
Explore simpler methods:
- average of each color channel
- first SVD component of each color channel
=#

Xmean = Vector{Matrix{Float32}}(undef, 3)
Xsvd = Vector{Matrix{Float32}}(undef, 3)
refold = v -> reshape(v, nx, ny)
for (i, c) in enumerate(channels) # separate color channels
    @info "channel $c"
    tmp_ = map(y -> getfield(y, c), Y3) # (nx,ny,nf)
    Xsvd[i] = refold(svd(reshape(tmp_, :, nf)).U[:,1]) # first component
    Xmean[i] = refold(mean(tmp_, dims=3))
end
Xmean = map(RGB{Float32}, Xmean...) # reassemble colors
L1 = Lpogm[:,:,1]
extrema(norm.(Lpogm .- Xmean))


#=
In this case
the temporal average of the video sequence
is pretty close to the low-rank component,
because this video is so simple.
The benefits of robust PCA
would be more apparent
with more complicated videos,
e.g.,
with illumination changes.
Even here one can see undesirable effects
of the sparse component
in the average
when we scale the difference image.
=#
pm = jim(
 jim(Xmean, "Average"),
 jim(L1, L"L_1"),
 jim(9 * abs.(L1 - Xmean), "9 × |L_1 - average|"),
## jim(L1 .== Xmean),
)

#=
Examining the first SVD component of each color
takes a bit more work.
SVD components have unit norm,
but RGB values should be in [0,1] range.
And the SVD has a sign ambiguity.
Even after correcting for those issues,
the SVD version has a somewhat color tint.
=#
jim(Xsvd; nrow=1, title="Xsvd before corrections", size=(600,200))

# Correct for SVD sign ambiguity
Xsvd = map(x -> x / sign(mean(x)), Xsvd)
jim(Xsvd; nrow=1, title="Xsvd after sign correction", size=(600,200))

# Correct for scaling
svdmax = maximum(maximum, Xsvd)
Xsvd = map(x -> x / svdmax, Xsvd)
Xsvd = map(RGB{Float32}, Xsvd...) # reassemble colors
ps = jim(
 jim(yf[1], "First frame"),
 jim(Lpogm[:,:,1], L"L_1"),
 jim(Xsvd, "Xsvd"),
 layout = (1,3),
 size = (600,200),
)


# Animate videos
anim1 = @animate for it in 1:nf
    tmp = stack([Y3[:,:,it], Lpogm[:,:,it], Spogm[:,:,it]])
    jim(tmp; nrow=1, title="Original | Low-rank | Sparse",
        xlabel = "Frame $it", size=(600, 250))
##  gui()
end
gif(anim1; fps = 6)

#src todo: compare pgm=ista, fpgm=fista, pogm


include("../../../inc/reproduce.jl")
