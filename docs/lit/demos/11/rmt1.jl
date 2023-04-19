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
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: rank, svd, svdvals, Diagonal, norm
using MIRTjim: prompt
using Plots: plot, scatter, scatter!, savefig, default
default(markerstrokecolor=:auto, label="", widen=true)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);



#=
## Finite-precision effects on singular values

Examine the singular values of a matrix
that ideally should have rank=1.
=#

n = 100
T = Float32
u = ones(T, n) / T(sqrt(n))
v = ones(T, n) / T(sqrt(n))
X = 1 * u * v' # theoretically rank-1 matrix
@assert rank(X) == 1 # julia is aware of precision limits
sigma = svdvals(X)
sigma[1]

#
xaxis = (L"k", (1,n), [1, n÷2, n])
p1 = scatter(sigma; xaxis, ylabel=L"σ_k", color=:blue)
good = findall(sigma .> 0)
xaxis = (L"k", (1,n), [2, n÷2, n])
p2 = scatter(good, log10.(sigma[good]);
    xaxis, ylabel=L"\log_{10}(σ_k)", color=:red)
plot(p1, p2, layout=(2,1))
#src savefig("round1.pdf")

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
X = u * v' # theoretically rank-1 matrix
rank(X) # 0 !

#src plot(py, )

#todo include("../../../inc/reproduce.jl")
