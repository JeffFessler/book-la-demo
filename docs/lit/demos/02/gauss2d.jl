#=
# [2d heatmap](@id gauss2d)

This example illustrates
matrix operations by making a 2D Gaussian plot
and computing area under a curve
and volume under a surface
using the Julia language.

- 2017-09-07, Jeff Fessler, University of Michigan
- 2023-06-06 Julia 1.9.0
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
        "MIRTjim"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using MIRTjim: jim
using Plots: default, heatmap, savefig
default(labelfontsize=18, tickfontsize=12, titlefontsize=18)


#=
## Broadcast
=#

x = range(-2, 2, 101)
y = range(-1.1, 1.1, 103) # deliberately non-square
A = abs2.(x) .+ 30 * abs2.(y)' # a lot is happening here!
F = exp.(-A)
p1 = heatmap(x, y, F', color=:grays, aspect_ratio=:equal)


#=
## Heatmap
Here is a fancy Julia way, now with labels:
=#
p2 = heatmap(range(-2, 2, 101), range(-1.1, 1.1, 103),
    (x,y) -> exp(-(abs2(x) + 3*abs2(y))), color=:grays, clim=(0,1),
    aspect_ratio=:equal, xlabel=L"x", ylabel=L"y", title=L"f(x,y)")

#=
## `jim`
The `jim` function from MIRTjim.jl has natural defaults.
=#
p3 = jim(x, y, F; xlabel=L"x", ylabel=L"y", title=L"f(x,y)", clim=(0,1))
## savefig(p3, "plot_exp4.pdf")


#=
## Area
Compute 1D integral ``\int_0^3 x^2 \, \mathrm{d}x`` numerically.
=#

f(x) = x^2 # parabola
x = range(0,3,2000) # sample points
w = diff(x) # "widths" of rectangles
Area = w' * f.(x[2:end])


#=
## Volume
2D integral
``\int_0^3 \int_0^2 \exp(-x^2 - 3 y^2) \, \mathrm{d}x \, \mathrm{d}y``
=#

f(x,y) = exp(-(x^2 + 3*y^2)) # gaussian bump function
x = range(0,3,2000) # sample points
y = range(0,2,1000) # sample points
w = diff(x) # "widths" of rectangles in x
u = diff(y) # "widths" of rectangles in y
F = f.(x[2:end], y[2:end]') # automatic broadcasting again!
Volume = w' * F * u

include("../../../inc/reproduce.jl")
