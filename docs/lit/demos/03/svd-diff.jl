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
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "InteractiveUtils"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LinearAlgebra: norm, I, diag, diagm, Diagonal
using MIRTjim: jim, prompt
using Plots; default(label="", markerstrokecolor=:auto, color=:blue)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && jim(:prompt, true);


#=
## First-order finite-difference matrix
=#

N = 16
Δ = diagm(0 => -ones(Int,N-1), 1 => ones(Int,N-1))[1:(N-1),:] # (N-1,N) diff
jim(Δ'; title="$(N-1) × $N finite-difference matrix Δ", color=:cividis)


# ## Left singular vectors are cos functions
h = π / N
v = k -> cos.(((1:N)*2 .- 1) * k * h / 2) / sqrt(N/2)

plot(v(5), line=:stem, marker=:circle, title="5th left singular vector",
 xtick=1:N, ylims=(-0.5,0.5), yticks=(-1:1)*0.5)

#
prompt()


# ## Right singular vectors are -sin functions
u = k -> -sin.((1:(N-1)) * k * h) / sqrt(N/2) # "derivative of cos is -sin"

plot(u(5), line=:stem, marker=:circle, title="5th right singular vector",
 xtick=1:N, ylims=(-0.5,0.5), yticks=(-1:1)*0.5)

#
prompt()


# ## Singular values
# ### (Caution: not in descending order)
σ = k -> 2*sin(k*h/2)
k = 1:(N-1)

scatter(k, σ.(k), title="$(N-1) singular values (unordered)",
 color=:red, xtick=1:N, ylims=(0,2), yticks=0:2, ywiden=true)

#
prompt()


# ## SVD components
V = hcat([v(k) for k in 1:(N-1)]...) # (N,N-1) "V_{N-1}" DCT
U = hcat([u(k) for k in 1:(N-1)]...) # (N-1,N-1) DST
Σ = Diagonal(σ.(1:(N-1))) # (N-1,N_1) Σ_N

jim(
 jim(V', "U: Left singular vectors"; color=:cividis),
 jim(U', "V: Right singular vectors"; color=:cividis),
 jim(Σ', "Σ: Singular values"; color=:cividis),
)


# ## Verify correctness of SVD

@assert all(>(0), diag(Σ)) # singular values are nonnegative
@assert Δ * V ≈ U * Σ # "derivative of cos is -sin"
@assert V'V ≈ I(N-1) # V is semi-unitary
@assert U'U ≈ I(N-1) && U*U' ≈ I(N-1) # U is unitary
@assert Δ ≈ U * Σ * V' # SVD of Δ


include("../../../inc/reproduce.jl")
