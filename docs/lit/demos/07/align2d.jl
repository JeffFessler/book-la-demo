#=
# [2D image alignment by rank-1 method](@id align2d)

This example illustrates 2D image alignment
via a simple 2D translation
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
        "Unitful"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using ImagePhantoms: SheppLoganEmis, spectrum, phantom
using ImagePhantoms: ellipse, ellipse_parameters
using LaTeXStrings
using LinearAlgebra: svd
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
Nx, Ny = 128, 128
Δx = FOV / Nx # pixel size
Δy = FOV / Ny
x = ((-Nx÷2):(Nx÷2-1)) * Δx
y = ((-Ny÷2):(Ny÷2-1)) * Δy
νx = ((-Nx÷2):(Nx÷2-1)) / Nx / Δx
νy = ((-Ny÷2):(Ny÷2-1)) / Ny / Δy

param1 = ellipse_parameters(SheppLoganEmis(), fovs=(FOV,FOV), disjoint=true)
param1 = param1[1:1]
shift = (1.7, 2.7) .* oneunit(Δx) # true non-integer shifts

param2 = [((p[1:2] .+ shift)..., p[3:end]...) for p in param1]
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


# Analytical spectra
data1 = spectrum(νx, νy, obj1)
data2 = spectrum(νx, νy, obj2)
ncps = @. data1 * conj(data2) / (abs(data1 * data2))# + eps())
fun = data -> log10.(abs.(data) / maximum(abs, data1))
psp = jim(
 jim(νx, νy, fun(data1), "|spectrum1|"),
 jim(νx, νy, fun(data2), "|spectrum2|"),
 jim(νx, νy, angle.(ncps); color = :hsv, title="phase difference")
)

#
prompt()


U, s, V = svd(ncps)
scatter(s, xlabel=L"k", ylabel=L"σ_k")

u = U[:,1]
v = V[:,1]
puv = plot(
 plot(νx, angle.(u), xlabel=L"νx"),
 plot(νy, angle.(v), xlabel=L"νy"),
)

#=
We could unwrap the phase and then fit a line
Instead we just take finite differences
and use `median` to eliminate the phase jumps.
=#

Δν = 1/FOV
myshift = (
 median(diff(angle.(u))) ./ (2π * Δν),
 median(diff(angle.(v))) ./ (2π * Δν),
)

@show shift
@show myshift

gui(); throw()



# Noisy data
seed!(0)
y = x0 + 2*randn(size(x0)); # noisy samples


## savefig(ps, "todo.pdf")


include("../../../inc/reproduce.jl")
