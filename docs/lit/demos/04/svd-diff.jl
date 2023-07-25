#=
# [SVD of finite differences](@id svd-diff)

This example illustrates the SVD of a first-order finite-difference matrix
using the Julia language.
This demo was inspired
by
[Gilbert Strang's 2006 article](https://archive.siam.org/news/news.php?id=828).
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
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: norm, I, diag, diagm, Diagonal
using MIRTjim: jim, prompt
using Plots: default, plot, scatter
default(); default(label="", markerstrokecolor=:auto, color=:blue, widen=:true)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


#=
## First-order finite-difference matrix
=#

N = 16
Δ = diagm(0 => -ones(Int,N-1), 1 => ones(Int,N-1))[1:(N-1),:] # (N-1,N) diff
jim(Δ'; title="$(N-1) × $N finite-difference matrix Δ", color=:cividis)


# ## Right singular vectors are cos functions
h = π / N
v = k -> cos.(((1:N)*2 .- 1) * k * h / 2) / sqrt(N/2)

plot(v(5), line=:stem, marker=:circle, title="5th right singular vector",
 xaxis = (L"i", (1,N), 1:N), yaxis=(L"v_5[i]", (-0.5,0.5), (-1:1)*0.5))

#
prompt()


# ## Left singular vectors are -sin functions
u = k -> -sin.((1:(N-1)) * k * h) / sqrt(N/2) # "derivative of cos is -sin"

plot(u(5), line=:stem, marker=:circle, title="5th left singular vector",
 xaxis = (L"i", (1,N), 1:N-1), yaxis=(L"u_5[i]", (-0.5,0.5), (-1:1)*0.5))

#
prompt()


# ## Singular values
# ### (Caution: not in descending order)
σ = k -> 2*sin(k*h/2)
k = 1:(N-1)

scatter(k, σ.(k), title="$(N-1) singular values (unordered)",
 color=:red, xaxis=(L"k", (1,N-1), 1:N-1), yaxis=(L"σ_k", (0,2), 0:2))

#
prompt()


# ## SVD components
V = hcat([v(k) for k in 1:(N-1)]...) # (N,N-1) "V_{N-1}" DCT
U = hcat([u(k) for k in 1:(N-1)]...) # (N-1,N-1) DST
Σ = Diagonal(σ.(1:(N-1))) # (N-1,N_1) Σ_N

jim(
 jim(U', "U: Left singular vectors"; color=:cividis),
 jim(V', "V: Right singular vectors"; color=:cividis),
 jim(Σ', "Σ: Singular values"; color=:cividis),
)


# ## Verify correctness of SVD

@assert all(>(0), diag(Σ)) # singular values are nonnegative
@assert Δ * V ≈ U * Σ # "derivative of cos is -sin"
@assert V'V ≈ I(N-1) # V is semi-unitary
@assert U'U ≈ I(N-1) && U*U' ≈ I(N-1) # U is unitary
@assert Δ ≈ U * Σ * V' # SVD of Δ


include("../../../inc/reproduce.jl")
