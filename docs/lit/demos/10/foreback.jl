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
default(); default(markerstrokecolor=:auto, label = "", markersize=6,
legendfontsize = 8) # todo: increase


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Load video data
=#

# Load raw data
y1 = [rand(RGB{Float16}, 48, 64) for it in 1:1499] # random test
if !@isdefined(y1)
    tmp = homedir() * "/111.mp4"
    if !isfile(tmp)
        url = "http://backgroundmodelschallenge.eu/data/synth1/111.mp4"
        @info "downloading 16MB from $url"
        tmp = download(url)
        @info "download complete"
    end
    y1 = VideoIO.load(tmp) # 1499 frames of size (480,640)
end;

# convert to arrays
if !@isdefined(Y3)
    tmp = y -> 1f0*permutedims((@view y[1:2:end,1:2:end]), (2,1)) # todo: downsample better
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
end

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
    niter = 20
    Xc = Array{Any}(undef, 3)
    out = Array{Any}(undef, 3)
    for (i, c) in enumerate(channels) # separate color channels
        @info "channel $c"
        Yc = map(y -> getfield(y, c), Y3);
        Xc[i], out[i] = robust_pca(Yc; α, β, mom = :pogm, niter)
    end
    Xpogm = map(RGB{Float32}, Xc...) # reassemble colors
end

#=
## Results
=#

# Extract low-rank (background) and sparse (foreground) components
Lpogm = Lpart(Xpogm)
Spogm = Spart(Xpogm)
iz = 81
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
