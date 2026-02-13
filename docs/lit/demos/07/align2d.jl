#=
# [2D image alignment by rank-1 method](@id align2d)

This example illustrates 2D image alignment
(estimating a simple 2D translation between an image pair)
using a rank-1 approximation
to the normalized cross power spectrum,
following the method of
[Hoge, IEEE T-MI, 2003](https://doi.org/10.1109/TMI.2002.808359),
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
        "ImagePhantoms"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
        "Statistics"
        "Unitful"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using ImagePhantoms: SheppLoganEmis, spectrum, phantom
using ImagePhantoms: ellipse, ellipse_parameters
using LaTeXStrings
using LinearAlgebra: norm, svd
using MIRTjim: jim, prompt
using Plots: default, gui, plot, plot!, scatter, scatter!, savefig
using Random: seed!
using Statistics: median
using Unitful: mm # use of physical units (mm here)
default(); default(label="", markerstrokecolor=:auto,
    guidefontsize=14, legendfontsize=14, tickfontsize=12)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Generate data

FOV = 256mm # physical units
Nx, Ny = 128, 126
Δx = FOV / Nx # pixel size
Δy = FOV / Ny
x = ((-Nx÷2):(Nx÷2-1)) * Δx
y = ((-Ny÷2):(Ny÷2-1)) * Δy
νx = ((-Nx÷2):(Nx÷2-1)) / Nx / Δx
νy = ((-Ny÷2):(Ny÷2-1)) / Ny / Δy;

# Ellipse parameters for 1st image:
param1 = ellipse_parameters(SheppLoganEmis(), fovs=(FOV,FOV))
shift = (1.7, 9.2) .* oneunit(Δx) # true non-integer shift

# Ellipse parameters for 2nd image:
param2 = [((p[1:2] .+ shift)..., p[3:end]...) for p in param1];

# Phantom images:
obj1 = ellipse(param1)
obj2 = ellipse(param2)
image1 = phantom(x, y, obj1)
image2 = phantom(x, y, obj2)
pim = jim(
 jim(x, y, image1, "image1"),
 jim(x, y, image2, "image2"),
 jim(x, y, image2-image1, "image2-image1")
)

#
prompt()


# Analytical spectra of these images (cf MRI)
spec1 = spectrum(νx, νy, obj1)
spec2 = spectrum(νx, νy, obj2);

# Add noise
seed!(0)
snr2sigma(db, y) = 10^(-db/20) * norm(y) / sqrt(length(y))
σnoise = snr2sigma(40, spec1)
addnoise(y) = y + σnoise * randn(ComplexF32, size(y))
data1 = addnoise(spec1)
data2 = addnoise(spec2);

# Normalized cross power spectrum
ncps = @. data1 * conj(data2) / (abs(data1 * data2));

# Show spectra and noisy phase difference
fun = data -> log10.(abs.(data) / maximum(abs, data1))
psp = jim(
 jim(νx, νy, fun(data1), "|spectrum1|"),
 jim(νx, νy, fun(data2), "|spectrum2|"),
 jim(νx, νy, angle.(ncps); color = :hsv, title="phase difference")
)

#
prompt()


#=
## SVD
By the shift property of the 2D Fourier transform,
the normalized cross power spectrum (NCPS) is
```math
e^{ı 2π (ν_x d_x + ν_y d_y)
=
e^{ı 2π ν_x d_x}
e^{ı 2π ν_y d_y}
```
where
``(d_x, d_y)``
is the 2D translation.

So the 2D NCPS
(in the absence of noise)
is an outer product
of 1D vectors,
each of which has the phase
associated with the translation
in one direction.

=#
U, s, V = svd(ncps)
psig = scatter(s, xlabel=L"k", ylabel=L"σ_k", title="Scree plot")

#
prompt()

# Phase of principal components
u = U[:,1]
v = conj(V[:,1]) # need conjugate here because of σ u v'
puv = plot(
 plot(νx, angle.(u), xlabel=L"ν_x"),
 plot(νy, angle.(v), xlabel=L"ν_y"),
)

#=
We could unwrap the phase and then fit a line
as suggested in the original paper.

Instead we just take finite differences
and use `median`
to eliminate the influence of the phase jumps.
=#

Δν = 1/FOV
myshift = (
 median(diff(angle.(u))) ./ (2π * Δν),
 median(diff(angle.(v))) ./ (2π * Δν),
)

# Error: the estimated shift is remarkably close to the true shift.
myshift .- shift

#
include("../../../inc/reproduce.jl")
