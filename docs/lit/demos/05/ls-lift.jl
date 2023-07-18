#=
# [LS lifting](@id ls-lift)

This example illustrates "lifting" in linear regression
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
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, svd
using MIRTjim: prompt
using Plots: default, gr, plotly, plot!, scatter, surface!, savefig
using Plots.PlotMeasures: px
using Random: seed!
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
 markersize = 6, tickfontsize=12, labelfontsize = 16, legendfontsize=16)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Noisy data
Samples of a nonlinear function.
=#
seed!(1) # seed rng
sfun = (t) -> 1 - cos(π/2*t)
M = 25
tm = sort(rand(M)) # M random sample locations
σ = 0.02
y = sfun.(tm) + σ * randn(M); # noisy samples

t0 = range(0, 1, 101) # fine sampling for showing curve
p1 = scatter(tm, y, color=:blue, label=L"\mathrm{data\ } y_m",
	xaxis = (L"t", (0, 1), 0:0.5:1),
	yaxis = (L"y_m", (-0.1, 1.1), 0:0.5:1),
)
plot!(t0, sfun.(t0), color=:black, label=L"s(t)", legend=:topleft)

#
prompt()


#=
## Polynomial fits
=#
Afun = (tt, deg) -> [t.^i for t in tt, i in 1:deg] # matrix of monomials

A1 = Afun(tm, 1) # M × 1 matrix
A2 = Afun(tm, 2) # M × 2 matrix

x1 = A1 \ y # LS solution for degree=1
plot!(p1, t0, Afun(t0,1)*x1, color=:red, label="linear model fit")

#
prompt()

x2 = A2 \ y # quadratic fit
plot!(p1, t0, Afun(t0,2)*x2, color=:orange, label="quadratic model fit")

#
prompt()

## savefig(p1, "04-ls-lift-1.pdf")


#=
## Lifting

We can view quadratic polynomial fitting
as nonlinear "lifting" from a 1D function of ``t``
to a 2D function of ``(t, t^2)``.
After such lifting,
regression with a linear model
fits much better,
as seen because the data points
nearly lie on the 2D plane.

Use `plotly()` backend here to view surface interactively.
=#

## plotly()
p2 = scatter(A2[:,1], A2[:,2], y, color=:blue, right_margin = 10px,
    xaxis = (L"t", (0,1), -1:1),
    yaxis = (L"t^2", (0,1), -1:1),
    zaxis = (L"y_m", (0,1), -1:1),
)
t1 = range(0, 1, 101)
t2 = range(0, 1, 102)
surface!(t1, t2, (t1,t2) -> x2[1]*t1 + x2[2]*t2, alpha=0.3)

#
prompt()

## gr(); # restore
## savefig(p2, "04-ls-lift-2.pdf") # with gr()


include("../../../inc/reproduce.jl")
