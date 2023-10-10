#=
# [LS fitting with cross validation](@id ls-cv)

This example illustrates least squares (LS) polynomial fitting,
with cross validation for selecting the polynomial degree,
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
        "MIRTjim"
        "Plots"
        "Polynomials"
        "Random"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using MIRTjim: prompt
using Plots: default, plot, plot!, scatter, scatter!, savefig
using Polynomials: fit
using Random: seed!
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
    markersize = 6, tickfontsize=12, labelfontsize = 16, legendfontsize=14)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Simulated data from latent nonlinear function
f(x) = 0.5 * exp(1.8 * x) # nonlinear function

seed!(0) # seed rng
M = 12 # how many data points
xm = sort(2*rand(M)) # M random sample locations
z = 0.5 * randn(M) # noise
y = f.(xm) + z # noisy samples

x0 = range(0, 2, 501) # fine sampling for showing curve
xaxis = (L"x", (0,2), 0:2)
yaxis = (L"y", (-2, 21), 0:4:20)
p0 = scatter(xm, y, color=:black, label="y (noisy data), M = $M"; xaxis, yaxis)
plot!(x0, f.(x0), color=:blue, label="f(x) : latent function", legend=:topleft)

#
prompt()
## savefig(p0, "ls-cv-data.pdf")


# ## Polynomial fitting

p1 = deepcopy(p0)
degs = [1, 3, 9]
for deg in degs
    pol = fit(xm, y, deg)
    plot!(p1, x0, pol.(x0), label = "degree $deg")
end
p1

#
prompt()
## savefig(p1, "ls-cv-fits.pdf")


# ## Over-fitting to noisy data

degs = 0:(M-1)
fits = zeros(length(degs))
accs = zeros(length(degs))
for (id, deg) in enumerate(degs)
    pol = fit(xm, y, deg)
    fits[id] = sqrt(sum(abs2, pol.(xm) - y))
    accs[id] = sqrt(sum(abs2, pol.(xm) - f.(xm)))
end
pf = scatter(degs, fits; color=:red,
 xaxis = ("degree", extrema(degs), [0,3,11]),
 yaxis = ("fits", (0,19), ),
 label = (L"‖ A_d \hat{x}_d - y ‖_2"),
)
scatter!(degs, accs;
 label = L"‖ A_d \hat{x}_d - f ‖_2", marker=:uptri, color=:blue)

prompt()
## savefig(pf, "ls-cv-over.pdf")


# ## Illustrate uncertainty

colors = [:red, :orange, :yellow, :green, :cyan, :blue, :grey, :black]
pols = Vector{Any}(undef, M)
deg1 = 8
p2 = plot(; xaxis, yaxis, title = "degree = $deg1",
 legend=:top)
scatter!(p2, [-9], [-9]; marker=:square, color=:gray, label="prediction")
scatter!(p2, [-9], [-9]; marker=:circle, color=:black, label="data")
for m in 1:M
    mm = (1:M)[[1:(m-1); (m+1):M]]
    pols[m] = fit(xm[mm], y[mm], deg1)
    color = colors[mod1(m, length(colors))]
    plot!(p2, x0, pols[m].(x0); xaxis, yaxis, title = "degree = $deg1",
     color,
    )
    pred = pols[m](xm[m])
    scatter!([xm[m]], [pred]; color, marker=:square)
    scatter!([xm[m]], [y[m]]; color=:black)
    plot!([1, 1]*xm[m], [y[m], pred]; color, line=:dash)
end

#
prompt()
## savefig(p2, "ls-cv-uq.pdf")


# ## Cross validation (leave-one-out)

degs = 1:8
errs = zeros(length(degs), M)
for (id, deg) in enumerate(degs)
    for m in 1:M
        mm = (1:M)[[1:(m-1); (m+1):M]]
        pol = fit(xm[mm], y[mm], deg)
        errs[id, m] = pol(xm[m]) - y[m]
    end
end
tmp = sqrt.(sum(abs2, errs, dims=2))

p3 = scatter(degs, tmp; legend = :top,
 xlabel = "degree",
 ylabel = "error",
 label = "Cross-validation loss",
)
scatter!(p3, degs, accs;
 label = L"‖ A_d \hat{x}_d - f ‖_2", marker=:uptri, color=:blue)

#
prompt()
## savefig(p3, "ls-cv-scat.pdf")

include("../../../inc/reproduce.jl")

