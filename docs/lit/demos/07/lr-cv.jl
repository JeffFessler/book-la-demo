#=
# [Low-Rank Selection via Cross Validation](@id lr-cv)

This example illustrates
low-rank matrix approximation
using cross-validation methods
for rank parameter selection,
using the Julia language.
As discussed by
[Owen & Perry, 2009](https://doi.org/10.1214/08-AOAS227),
separate row or column hold-out
is ineffective,
whereas
[Bi-Cross-Validation (BCV)](https://doi.org/10.1214/08-AOAS227)
is more effective.
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
using Statistics: mean
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
scatter!(pk, ranks, cv, color=:green, marker=:star,
    label=L"\mathrm{BCV}",
)
i_bcv = argmin(cv)
scatter!([ranks[i_bcv]], [cv[i_bcv]], color=:black, marker=:star, markersize=4,)

#
prompt()

## savefig(psk, "lr_bcv1.pdf")


#=
Compare with row or column hold-out CV
=#

"""
    function lr_cross_validation_by_column(Y, fold, n_components)
"""
function lr_cross_validation_by_column(
    X::AbstractMatrix{<:Number},
    fold::Int,
    n_components::AbstractVector{<:Int},
)

    n_samples = size(X, 2) # Assuming columns are samples
    fold_size = n_samples ÷ fold
    errors = zeros(length(n_components), fold)

    for fold_idx in 1:fold
        # Define train and test indices for this fold
        test_indices = ((fold_idx - 1) * fold_size + 1):min(fold_idx * fold_size, n_samples)
        train_indices = setdiff(1:n_samples, test_indices)

        X_train = X[:, train_indices]
        X_test = X[:, test_indices]

        U, _, _ = svd(X_train) # "PCA" of training data

        for (comp_idx, n_component) in enumerate(n_components)
            Ur = U[:,1:n_component]
            X_test_reconstructed = Ur * (Ur' * X_test)
            errors[comp_idx, fold_idx] = # calculate reconstruction error
                norm(X_test - X_test_reconstructed) / norm(X_test)
        end
    end
    return errors * 100
end;

#=
## Apply elementary CV to same noisy data

Holding out rows or columns
leads to highly over-estimated ranks,
as predicted in the literature.

This is the approach recommended by GPT 4.1 (circa 2025-08),
presumably because holding out individual data points
is prevalent in machine learning.
=#

fold = 5
Kmax = min(M,N)÷fold
n_components = 0:Kmax
errors_by_col = lr_cross_validation_by_column(Y, fold, n_components)
error_means_by_col = vec(mean(errors_by_col, dims=2))
i_col = argmin(error_means_by_col) # best based on minimum mean error

errors_by_row = lr_cross_validation_by_column(Y', fold, n_components)
error_means_by_row = vec(mean(errors_by_row, dims=2));
i_row = argmin(error_means_by_row) # best based on minimum mean error

optimal_k_col = n_components[i_col]
optimal_k_row = n_components[i_row]

pcv = plot(
 xlims=(0,10),
 xticks=[0, 1, 5, 10],
 ylims=(0,100),
 widen = true,
 xlabel = "rank",
 ylabel = "NRMSD",
)
scatter!(n_components, error_means_by_col, label="by column")
scatter!(n_components, error_means_by_row, label="by row", marker=:x)
scatter!([n_components[i_col]], [error_means_by_col[i_col]],
 color=:black, marker=:circle, markersize=4, )
scatter!([n_components[i_row]], [error_means_by_row[i_row]],
 color=:black, marker=:x, markersize=4, )

include("../../../inc/reproduce.jl")
