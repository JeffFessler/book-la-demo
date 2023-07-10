#=
# [Wavelet frame denoising](@id frame-cycle)

This example illustrates
image denoising
using a frame defined by a combination
of an orthonormal discrete wavelet transform (ODWT)
and "cycle-spinning" operators,
using the Julia language.
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
        "LinearMapsAA"
        "MIRT"
        "MIRTjim"
        "Plots"
        "Random"
        "Statistics"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ImagePhantoms: circle, phantom
using InteractiveUtils: versioninfo
## using LaTeXStrings
using LinearAlgebra: norm
using LinearMapsAA: LinearMapAA
using MIRT: Aodwt #, pogm_restart
using MIRTjim: jim, prompt
using Plots: default, gui, plot, savefig
using Random: seed!
default(); default(markerstrokecolor=:auto, label = "", markersize=6,
 tickfontsize = 9, labelfontsize = 16, titlefontsize = 16)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Generate test image
Both clean and noisy image.
=#
if !@isdefined(ydata)
    nx, ny = 144, 128 # multiples of 2^3 for wavelet code
    ob = circle(40f0, 6f0)
    x = (1:nx) .- (nx-1)/2
    y = (1:ny) .- (ny-1)/2
    oversample = 3
    xtrue = phantom(x, y, [ob], oversample)
    seed!(0)
    ydata = xtrue + 0.7f0 * randn(Float32, nx, ny)
    nrmse = (xh) -> round(norm(xh - xtrue) / norm(xtrue) * 100, digits=1)
end;

clim = (-2, 8)
clim = (-1, 7)
py = jim(
 jim(xtrue; clim, title="True Image"),
 jim(ydata; clim, xlabel="NRMSE=$(nrmse(ydata))%", title="Noisy Image"),
 size = (800, 350),
)
## savefig(py, "frame-cycle-py.pdf")


# Orthogonal discrete wavelet transform operator (`LinearMapAO`):
W, scales, _ = Aodwt((nx,ny) ; T = eltype(ydata), level = 3)
scales = Int.(scales)
isdetail = scales .> 0
pw = jim(
   jim(scales, "wavelet scales"; color=:viridis),
   jim(real(W * xtrue) .* isdetail, "wavelet detail coefficients";
       color=:cividis),
   size = (800, 320),
)



#=
## ODTW denoising
This simply uses soft thresholding,
the proximal operator for the 1-norm.

Define proximal operator so that it shrinks only the detail coefficients:
=#

soft = (z,c) -> sign(z) * max(abs(z) - c, 0) # soft thresholding
reg = 0.9 # hand-tuned for small NRMSE
g_prox = (z,c) -> soft.(z, isdetail .* (reg * c))

## Apply wavelet coefficient soft thresholding
coef = W * ydata
xhat1 = W' * g_prox(coef, 1)
jim(coef)
p1 = jim(xhat1; clim, xlabel="NRMSE=$(nrmse(xhat1))%", title="ODWT denoised")

#=
The NRMSE is reduced substantially,
but there are severe "block" artifacts
due to the dyadic decomposition
of the ODWT.
=#

#=
## Frame approach
Define a frame based on combining ODWT with ``K`` `circshift` operations.
The analysis operator is
```math
\mathbf{T} = \frac{1}{\sqrt{K}}
\begin{bmatrix}
\mathbf{W} \mathbf{P}_1 \\
\mathbf{W} \mathbf{P}_2 \\ \vdots \\
\mathbf{W} \mathbf{P}_K
\end{bmatrix}
```
where ``\mathbf{P}_0 = \mathbf{I}``
and
where each ``\mathbf{P}_k``
is a `circshift` operator.
=#

## Define circshift permutation map
Pforw = shifts -> (x -> circshift(x, shifts))
Pback = shifts -> (y -> circshift(y, -1 .* shifts))
Pmap = shifts -> LinearMapAA(Pforw(shifts), Pback(shifts), (nx*ny,nx*ny); 
    odim=(nx,ny), idim=(nx,ny), T=Float32, prop=(; shifts, name="shift"))

p12 = Pmap((1,2))
@assert p12' * (p12 * ydata) ≈ ydata # check Pmap

## tmp = W * p12
## tmp * xtrue # todo fails!?
## Top = vcat([W * Pmap((xs,ys)) for xs in shifts, ys in shifts]...)

## Define Parseval tight frame analysis operator
shifts = -3:3
Pmaps = [Pmap((xs,ys)) for xs in shifts, ys in shifts]
K = length(Pmaps)
Tforw = x -> stack(k -> (W * (Pmaps[k] * x)) / sqrt(K), 1:K, dims=3)
Tback = y -> sum(k -> Pmaps[k]' * (W' * y[:,:,k]), 1:K) / sqrt(K)
Top = LinearMapAA(Tforw, Tback, (nx*ny*K, nx*ny); 
    odim=(nx,ny,K), idim=(nx,ny), T=Float32, prop=(; name="Top"))

## Sanity check that the operator satisfies the tight frame condition:
@assert Top' * (Top * ydata) ≈ ydata


#=
## Parseval tight frame (PTF) denoising

The tight frame approach
leads to lower NRMSE
and reduces the block artifacts.

todo: describe cost functions and implement POGM
=#
xhat2 = Top' * g_prox(Top * ydata, 0.2) # todo: hand-tuned again
p2 = jim(xhat2; clim, xlabel="NRMSE=$(nrmse(xhat2))%", title="PTF denoised")
pf = jim(p1, p2; size=(800,350))

## savefig(pf, "frame-cycle-pf.pdf")


include("../../../inc/reproduce.jl")
