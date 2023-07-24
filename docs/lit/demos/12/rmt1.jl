#=
# [Random matrix theory and rank-1 signal + noise](@id rmt)

This example compares results from random matrix theory
with empirical results
for rank-1 matrices with additive noise 
using the Julia language.
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
        "MIRTjim"
        "Plots"
        "StatsBase"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: rank, svd, svdvals, Diagonal, norm
using MIRTjim: prompt
using Plots: default, gui, plot, plot!, scatter, scatter!, savefig, histogram
using StatsBase: mean, var
default(markerstrokecolor=:auto, label="", widen=true, markersize = 6,
 tickfontsize = 12, labelfontsize = 18,
)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

#todo isinteractive() && prompt(:prompt);



#=
## Finite-precision effects on singular values

Examine the singular values of a matrix
that ideally should have rank=1.
=#

n = 100
T = Float32
u = ones(T, n) / T(sqrt(n))
v = ones(T, n) / T(sqrt(n))
Y = 1 * u * v' # theoretically rank-1 matrix
@assert rank(Y) == 1 # julia is aware of precision limits
sigma = svdvals(Y)
sigma[1], abs(sigma[1] - 1)

#
xaxis = (L"k", (1,n), [1, n÷2, n])
p1 = scatter(sigma; xaxis, yaxis = (L"σ_k", (-0.02, 1.02), -1:1), color=:blue)
good = findall(sigma .> 0)
xaxis = (L"k", (1,n), [2, n÷2, n])
p2 = scatter(good, log10.(sigma[good]);
    xaxis, yaxis = (L"\log_{10}(σ_k)", (-40, 2), -40:20:0), color=:red)
p12 = plot(p1, p2, layout=(2,1))
#src savefig(p12, "round1.pdf")

#
prompt()


#=
## Finite-precision effects on rank

For a matrix that is sufficiently large
relative to the precision of the matrix elements,
the threshold in the `rank` function
can be high enough
that the returned rank is `0`.
=#

using LinearAlgebra: rank, svdvals
n = 1100 # > 1 / eps(Float16)
T = Float16
u = ones(T, n) / T(sqrt(n))
v = ones(T, n) / T(sqrt(n))
Y = u * v' # theoretically rank-1 matrix
rank(Y) # 0 !


N = 100
X = (2rand(BigFloat, N, N).-1)*sqrt(3) # very high-precision reference, var=1
for T in (Float16, Float32, Float64)
    local Y = T.(X) # quantize
    local Z = T.(Y - X) # quantization error
    @show T, mean(Z) # approximately 0
    vr = var(Float64.(Z)) # sample variance
    @show (vr, eps(T)^2) # empirical, predicted
end

T = Float32
Y = T.(X) # quantize
Z = T.(Y - X) # quantization error

# Examine histogram of floating point quantization errors
ph = histogram(vec(Z), bins = (-40:40)/40 * eps(T))


# Examine floating point quantization errors
x = range(BigFloat(-1), BigFloat(1), 1001) * 2
z = T.(x) - x # quantization error
ylabel = latexstring("error: \$\\ (q(x) - x)/ϵ\$")
scatter(x, z / eps(T), yaxis=(ylabel, (-1,1).*0.51, (-2:2)*0.25))
plot!(x, (@. eps(T(x)) / eps(T) / 2), label=L"ϵ/2", color=:blue)
pq = plot!(x, x/2, xaxis=(L"x",), label=L"x/2", legend=:top, color=:red)

#=
Based on the quantization error plot above,
the quantization error for a floating point number near ``x``
is bounded above by ``ϵ x / 2``.
Thus if ``x ∼ U(-a,a)`` then
``
E[z^2] = E[|q(x) - x|^2]
= \frac{1}{2a} ∫_{-a}^a |q(x) - x|^2 \mathrm{d}x
\leq
\frac{1}{a} ∫_0^a |ϵ x / 2|^2 \mathrm{d}x
= ϵ^2 a^2 / 12.
``
=#

#src plot(py, )

#todo include("../../../inc/reproduce.jl")
