#=
# [Low-Rank via Cross Validation](@id lr-cv)

This example illustrates
low-rank matrix approximation
using
[Bi-Cross-Validation (BCV)](https://doi.org/10.1214/08-AOAS227)
for rank parameter selection,
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
using LinearAlgebra: svd, svdvals, Diagonal, norm, pinv
using MIRTjim: prompt
using Plots: default, gui, plot, plot!, scatter!, savefig
using Random: seed!, randperm
default(); default(label="", markerstrokecolor=:auto, markersize=7,
    labelfontsize=20, tickfontsize=16, legendfontsize=17, widen=true)



# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Generate data

# Noiseless low-rank matrix and noisy data matrix

M, N = 100, 50 # problem size
seed!(0)
Ktrue = 5 # true rank (planted model)
X = svd(randn(M,Ktrue)).U * Diagonal(1:Ktrue) * svd(randn(Ktrue,N)).Vt
sig0 = 0.03 # noise standard deviation
Y = X + sig0 * randn(size(X)) # noisy
sy = svdvals(Y)
sx = svdvals(X)
sx[1:Ktrue]

#
sy[1:Ktrue]

# ### Plot singular values
ps = plot(xaxis = (L"k", (1,N), [1, Ktrue, N]), yaxis = (L"σ", (0,5.5), 0:5))
scatter!(1:N, sy, color=:red, marker=:hexagon,
 label=L"\sigma_k(Y) \ \mathrm{noisy}")
scatter!(1:N, sx, color=:blue, label=L"\sigma_k(X) \ \mathrm{noiseless}")

#
prompt()

## savefig(ps, "lr_sure1s.pdf")


# ## Low-rank approximation with various ranks
(U, sy, V) = svd(Y)
nrmse_K = zeros(N)
nrmsd_K = zeros(N)
nrmsd = (D) -> norm(D) / norm(Y) * 100
nrmse = (D) -> norm(D) / norm(X) * 100
for K in 1:N
    Xh = U[:,1:K] * Diagonal(sy[1:K]) * V[:,1:K]'
    nrmsd_K[K] = nrmsd(Xh - Y)
    nrmse_K[K] = nrmse(Xh - X)
end
nrmsd_K = [nrmsd(0 .- Y); nrmsd_K]
nrmse_K = [nrmse(0 .- X); nrmse_K]
klist = 0:N;


# ### Plot normalized root mean-squared error/difference versus rank K
pk = plot( # legend=:outertop,
    xaxis = (L"K", (1,N), [0, 2, Ktrue, N]),
    yaxis = ("'Error' [%]", (0, 100), 0:20:100),
)
scatter!(klist, nrmse_K, color=:blue,
    label=L"\mathrm{NRMSE\ } ‖ \! \hat{X}_K - X \ ‖_{\mathrm{F}} / ‖X \ ‖_{\mathrm{F}} \cdot 100\%",
)
scatter!(klist, nrmsd_K, color=:red, marker=:diamond,
    label=L"\mathrm{NRMSD\ } ‖ \! \hat{X}_K - Y \ ‖_{\mathrm{F}} / ‖Y \ ‖_{\mathrm{F}} \cdot 100\%",
)

#
prompt()

## savefig(pk, "lr_sure1a.pdf")


# ### Bi-cross-validation code

"""
    bcv(Y::AbstractMatrix{<:Number}, ranks=1:10)
Compute bi-cross-validation per
https://doi.org/10.1214/08-AOAS227
"""
function bcv(Y::AbstractMatrix{<:Number}, ranks=1:10, fold::Int=2)
    M, N = size(Y)
    any(>(min(M,N)), ranks) && throw("bad ranks")
    any(<(0), ranks) && throw("bad ranks")
    H1 = M÷fold # hold-out rows
    H2 = N÷fold # hold-out columns
    perm1 = randperm(M)
    hold1 = perm1[1:H1]
    keep1 = perm1[(H1+1):M]
    perm2 = randperm(N)
    hold2 = perm2[1:H2]
    keep2 = perm2[(H2+1):N]
    A = Y[hold1,hold2]
    B = Y[hold1,keep2]
    C = Y[keep1,hold2]
    D = Y[keep1,keep2]
    U,s,V = svd(D)
    error = zeros(length(ranks))
    for (i, r) in enumerate(ranks)
        Dr_pinv = V[:,1:r] * Diagonal(pinv.(s[1:r])) * U[:,1:r]'
        error[i] = norm(A - B * Dr_pinv * C)
    end
    return error / norm(A) * 100
end;


#=
### Apply BCV to synthetic data

In this example, (2×2)-fold BCV
is minimized at the correct rank of 5.
=#

fold = 2
ranks = 0:min(M,N)÷fold
cv = bcv(Y, ranks, fold)
scatter!(pk, ranks, cv, color=:green, marker=:start,
    label=L"\mathrm{BCV}",
)

#
prompt()

## savefig(psk, "lr_bcv1.pdf")


include("../../../inc/reproduce.jl")
