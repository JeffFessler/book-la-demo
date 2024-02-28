#=
# [Roundoff errors and rank](@id rmt-round1)

This example examines the effects of
[roundoff error](https://en.wikipedia.org/wiki/Round-off_error)
associated with finite-precision arithmetic
on matrix `rank` and singular value calculations,
using the Julia language.
The focus is rank-1 matrices.
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
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "StatsBase"
    ])
end


# Tell Julia to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: rank, svd, svdvals, Diagonal, norm
using MIRTjim: prompt
using Plots: default, gui, plot, plot!, scatter, scatter!, savefig, histogram
using Plots.PlotMeasures: px
using StatsBase: mean, var
default(markerstrokecolor=:auto, label="", widen=true, markersize = 6,
 tickfontsize = 12, labelfontsize = 18, legendfontsize = 18, linewidth=2,
)


# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);



#=
## Roundoff errors and singular values

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
## savefig(p12, "round1.pdf")

#
prompt()


#=
## Roundoff errors and rank

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

#=
The following plot shows the singular values
and the threshold set in the `rank` function.
There is a gap with a ratio of ``10^5`` between
``σ₁ = 1``
and
``σ₂``
so `rank`'s threshold is unnecessarily conservative in this case.

RMT suggests that the (smaller) tolerance
``\max(M,N) \, ε \, σ₁``
can suffice,
and indeed in this particular example
it correctly separates the nonzero signal singular value
from the noise singular values.

Here the matrix `Y` nicely fits the assumptions of RMT;
there may be other situations
with "worst case" data matrices
where the conservative threshold is needed.

We have noticed that this plot looks different
on a Mac and on Linux with Julia 1.9.2.
Apparently some differences in the SVD libraries
on different systems can affect the details
of the tiny singular values.
=#
s = svdvals(Y)
tol = minimum(size(Y)) * eps(T) * s[1] # from rank()
tol2 = sqrt(maximum(size(Y))) * eps(T) * s[1] # from RMT
p16 = plot([1, n], [1,1] * log10(tol),
 label="rank threshold: tol=$(round(tol,digits=2))",
 title = "Rank-1 matrix with $T elements",
)
plot!([1, n], [1,1] * log10(tol2),
 label="RMT threshold: tol=$(round(tol2,digits=2))")
scatter!(1:n, log10.(s); label="singular values", alpha=0.8,
 xaxis = (L"k", (1,n), [2, n÷2, n]),
 yaxis = (L"\log_{10}(σ)", (-45, 2), [-40:20:0; -5]),
 left_margin = 40px, bottom_margin = 20px,
 annotate = (200, -8, Sys.MACHINE, :left),
)

#
prompt()
## savefig(p16, "round1-p16-$(Sys.MACHINE).pdf")


#=
## Roundoff error variance
=#
N = 200
# very high-precision reference, var=1
X = (2 * rand(BigFloat, N, N) .- 1) * sqrt(3)
for T in (Float16, Float32, Float64)
    local Y = T.(X) # quantize
    local Z = T.(Y - X) # quantization error
    @show T, mean(Z) # approximately 0
    vr = var(Float64.(Z)) # sample variance
    @show (vr, eps(T)^2/24) # empirical, predicted
end


#=
## Roundoff error plot
=#

T = Float32
Y = T.(X) # quantize
Z = T.(Y - X); # quantization error

# Examine histogram of floating point quantization errors
ph = histogram(vec(Z), bins = (-40:40)/40 * eps(T))

#
prompt()

# Examine floating point quantization errors
x = range(BigFloat(-1), BigFloat(1), 1001) * 2
z = T.(x) - x # quantization error
ylabel = latexstring("error: \$\\ (q(x) - x)/ϵ\$")
scatter(x, z / eps(T), yaxis=(ylabel, (-1,1).*0.51, (-2:2)*0.25))
plot!(x, (@. eps(T(x)) / eps(T) / 2), label=L"ϵ(x)/2", color=:blue)
pq = plot!(x, x/2, xaxis=(L"x",), label=L"x/2", legend=:top, color=:red)

#
prompt()


#=
Based on the quantization error plot above,
the quantization error for a floating point number near ``x``
is bounded above by ``ϵ x / 2``.
See the
[Julia manual for `eps`](https://docs.julialang.org/en/v1/base/base/\#Base.eps-Tuple{AbstractFloat}).
Thus if ``x ∼ \mathrm{Unif}(-a,a)`` then
``
E[z^2] = E[|q(x) - x|^2]
= \frac{1}{2a} ∫_{-a}^a |q(x) - x|^2 \mathrm{d} x
\leq
\frac{1}{a} ∫_0^a |ϵ x / 2|^2 \mathrm{d} x
= ϵ^2 a^2 / 12.
``
=#


include("../../../inc/reproduce.jl")
