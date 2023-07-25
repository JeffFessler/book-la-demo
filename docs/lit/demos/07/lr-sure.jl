#=
# [Low-Rank SURE](@id lr-sure)

This example illustrates
Stein's unbiased risk estimation (SURE)
for parameter selection
in low-rank matrix approximation,
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
using LinearAlgebra: svd, svdvals, Diagonal, norm
using MIRTjim: prompt
using Plots: default, gui, plot, plot!, scatter!, savefig
using Random: seed!
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


# ## Explore (nuclear norm) regularized version
soft = (s,β) -> max.(s-β,0) # soft threshold function
dsoft = (s,β) -> Float32.(s .> β) # "derivative" thereof
reglist = [range(0, 0.5, 11); 0.75:0.25:6]
Nr = length(reglist)
nrmse_reg = zeros(Nr)
nrmsd_reg = zeros(Nr)
for ir in 1:Nr
    reg = reglist[ir]
    Xh = U * Diagonal(soft.(sy,reg)) * V'
    nrmsd_reg[ir] = nrmsd(Xh - Y)
    nrmse_reg[ir] = nrmse(Xh - X)
end;


# ### Plot NRMSE and NRMSD versus regularization parameter
pb = plot(legend=:topleft, xaxis = (L"β", (0,6), 0:6),
    yaxis = ("'Error' [%]", (0, 100), 0:20:100))
scatter!(reglist, nrmse_reg, color=:blue,
    label=L"\mathrm{NRMSE\ } ‖ \! \hat{X}_{\beta} - X \ ‖_{\mathrm{F}} / ‖X \ ‖_{\mathrm{F}} \cdot 100\%",
##  label=L"\mathrm{NRMSE}", # book
)
scatter!(reglist, nrmsd_reg, color=:red, marker=:diamond,
    label=L"\mathrm{NRMSD\ } ‖ \! \hat{X}_{\beta} - Y \ ‖_{\mathrm{F}} / ‖Y \ ‖_{\mathrm{F}} \cdot 100\%",
##  label=L"\mathrm{NRMSD}", # book
)

#
prompt()

## savefig(pb, "lr_sure1b.pdf")


#=
## Explore SURE for selecting ``β``

```math
\mathrm{SURE}(β) = ‖ \hat{X} - Y ‖^2 - MN \sigma_0^2
 + 2 σ_0^2 \left( |M - N| \sum_{i=1}^{\min(M,N)} \frac{h(σ_iσ)}{σ_i}
 + \sum_{i=1}^{\min(M,N)} \dot{h}_i(σ_i;β)
 + 2 \sum_{i \neq j}^{\min(M,N)} \frac{σ_i h_i(σ_i;β)}{σ_i^2 - σ_j^2} \right)
```

- `sy` singular values of Y
- `reg` regularization parameter
- `v0 = sigma_0^2` noise variance
=#
function sure(sy, reg, v0, M, N)
    sh = soft.(sy, reg) # estimated singular values
    big = sy.^2 .- (sy.^2)'
    big[big .== 0] .= Inf # trick to avoid divide by 0
    big = (sy .* sh) ./ big # [sy[i] * sh[i] / big[i,j] for i in 1:N, j in 1:N]
    big = sum(big)
    norm(sh - sy)^2 - M*N*v0 + 2*v0*(abs(M-N)*sum(sh ./ sy) + sum(dsoft.(sy,reg)) + 2*big)
end

# ### Evaluate SURE for each candidate regularization parameter
sure_reg = [sure(sy, reglist[ir], sig0^2, M, N) for ir in 1:Nr]
reg_best = reglist[argmin(sure_reg)] # SURE pick for β


# ### Plot NRMSE and NRMSD versus regularization parameter
psb = plot(legend=:bottomright, widen=true,
    xaxis = (L"β", (0,6), [reg_best, 5, 6]),
    yaxis = ("'Error' [%]", (0,100), 0:20:100),
)
scatter!(reglist, nrmse_reg, color=:blue,
    label=L"\mathrm{NRMSE\ } ‖ \! \hat{X}_\beta - X \ ‖_{\mathrm{F}} / ‖X \ ‖_{\mathrm{F}} \cdot 100\%",
##  label=L"\mathrm{NRMSE}", # book
)
scatter!(reglist, nrmsd_reg, color=:red, marker=:diamond,
    label=L"\mathrm{NRMSD\ } ‖ \! \hat{X}_\beta - Y \ ‖_{\mathrm{F}} / ‖Y \ ‖_{\mathrm{F}} \cdot 100\%",
##  label=L"\mathrm{NRMSD}", # book
)
scatter!(reglist, sqrt.(sure_reg)/norm(Y)*100, color=:green, marker=:star,
    label=L"(\mathrm{SURE}(\beta))^{1/2} / ‖Y \ ‖_{\mathrm{F}} \cdot 100\%")

#
prompt()

## savefig(psb, "lr_sure1c.pdf")


# ### Examine shrunk singular values for best regularization parameter
sh = soft.(sy,reg_best)
psk = plot(
    xaxis = (L"k", (1, N), [1, Ktrue, sum(sh .!= 0), N]),
    yaxis = (L"σ", (0, 5.5), 0:6),
    legendfontsize = 20,
)
scatter!(1:N, sy, color=:red, marker=:hexagon, label=L"\sigma(Y) \ \mathrm{noisy}")
scatter!(1:N, sx, color=:blue, label=L"\sigma(X) \ \mathrm{noiseless}")
scatter!(1:N, sh, color=:green, marker=:star, label=L"\hat{\sigma} \ \ \mathrm{SURE} \ \hat{\beta}")

#
prompt()

## savefig(psk, "lr_sure1t.pdf")


include("../../../inc/reproduce.jl")
