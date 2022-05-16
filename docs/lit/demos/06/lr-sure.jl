#---------------------------------------------------------
# # [Low-Rank SURE](@id lr-sure)
#---------------------------------------------------------

#=
This example illustrates
Stein's unbiased risk estimation (SURE)
for parameter selection
in low-rank matrix approximation,
using the Julia language.
=#

#=
This entire page was generated using a single Julia file:
[lr-sure.jl](@__REPO_ROOT_URL__/06/lr-sure.jl).
=#
#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](http://nbviewer.jupyter.org/) here:
#md # [`lr-sure.ipynb`](@__NBVIEWER_ROOT_URL__/06/lr-sure.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`lr-sure.ipynb`](@__BINDER_ROOT_URL__/06/lr-sure.ipynb),

#=
First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: svd, svdvals, Diagonal, norm
using Random: seed!
using Plots; default(label="", markerstrokecolor=:auto, markersize=7,
 guidefontsize=13, tickfontsize=12, legendfontsize=13, widen=true)
using LaTeXStrings
using MIRTjim: prompt
using InteractiveUtils: versioninfo



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
plot(xlabel=L"k", ylabel=L"\sigma")
scatter!(1:N, sy, color=:blue, label=L"\sigma_k(Y) \ \mathrm{noisy}")
scatter!(1:N, sx, color=:red, label=L"\sigma_k(X) \ \mathrm{noiseless}")
plot!(xtick=[1, Ktrue, N], ytick=0:5, xlim=(1,N), ylim=(0, 5.5))

#
prompt()

#src savefig("06_lr_sure1s.pdf")


# ## Low-rank approximation with various ranks
(U, sy, V) = svd(Y)
nrmse_K = zeros(N)
nrmsd_K = zeros(N)
nrmsd = (D) -> norm(D) / norm(Y) * 100
nrmse = (D) -> norm(D) / norm(X) * 100
for K=1:N
    Xh = U[:,1:K] * Diagonal(sy[1:K]) * V[:,1:K]'
    nrmsd_K[K] = nrmsd(Xh - Y)
    nrmse_K[K] = nrmse(Xh - X)
end
nrmsd_K = [nrmsd(0 .- Y); nrmsd_K]
nrmse_K = [nrmse(0 .- X); nrmse_K]
klist = 0:N;


# ### Plot normalized root mean-squared error/difference versus rank K
plot(xtick=[0, 2, Ktrue, N], ytick=0:20:100,
    xlabel=L"K", ylabel="'Error' [%]", xlim=(1,N), ylim=(0, 100))
scatter!(klist, nrmse_K, color=:blue,
    label=L"\mathrm{NRMSE\ } \|\hat{X}_K - X\|_F / \|X\|_F \cdot 100\%")
scatter!(klist, nrmsd_K, color=:red,
    label=L"\mathrm{NRMSD\ } \|\hat{X}_K - Y\|_F / \|Y\|_F \cdot 100\%")

#
prompt()

#src savefig("06_lr_sure1a.pdf")


# ## Explore (nuclear norm) regularized version
soft = (s,β) -> max.(s-β,0) # soft threshold function
dsoft = (s,β) -> Float32.(s .> β) # "derivative" thereof
reglist = [LinRange(0, 1, 20); 1:0.25:6]
Nr = length(reglist)
nrmse_reg = zeros(Nr)
nrmsd_reg = zeros(Nr)
for ir=1:Nr
    reg = reglist[ir]
    Xh = U * Diagonal(soft.(sy,reg)) * V'
    nrmsd_reg[ir] = nrmsd(Xh - Y)
    nrmse_reg[ir] = nrmse(Xh - X)
end;


# ### Plot NRMSE and NRMSD versus regularization parameter
plot(xtick=0:6, ytick=0:20:100, legend=:bottomright,
    xlabel=L"\beta", ylabel="'Error' [%]", xlim=(0,6), ylim=(0, 100))
scatter!(reglist, nrmse_reg, color=:blue,
    label=L"\mathrm{NRMSE\ } \|\hat{X}_{\beta} - X\|_F / \|X\|_F \cdot 100\%")
scatter!(reglist, nrmsd_reg, color=:red,
    label=L"\mathrm{NRMSD\ } \|\hat{X}_{\beta} - Y\|_F / \|Y\|_F \cdot 100\%")

#
prompt()

#src savefig("06_lr_sure1b.pdf")


# ## Explore SURE for selecting $\beta$
#
# $SURE(\beta) = \Vert \hat{X} - Y \Vert^2 - MN \sigma_0^2 + 2 \sigma_0^2 \left( |M - N| \sum_{i=1}^{\min(M,N)} \frac{h(\sigma_i;\beta)}{\sigma_i} + \sum_{i=1}^{\min(M,N)} \dot{h}_i(\sigma_i;\beta) + 2 \sum_{i \neq j}^{\min(M,N)} \frac{\sigma_i h_i(\sigma_i;\beta)}{\sigma_i^2 - \sigma_j^2} \right) $

# sy: singular values of Y
# reg: regularization parameter
# v0 = sigma_0^2 noise variance
function sure(sy, reg, v0, M, N)
    sh = soft.(sy, reg) # estimated singular values
    big = sy.^2 .- (sy.^2)'
    big[big .== 0] .= Inf # trick to avoid divide by 0
    big = (sy .* sh) ./ big # [sy[i] * sh[i] / big[i,j] for i=1:N, j=1:N]
    big = sum(big)
    norm(sh - sy)^2 - M*N*v0 + 2*v0*(abs(M-N)*sum(sh ./ sy) + sum(dsoft.(sy,reg)) + 2*big)
end

# ### Evaluate SURE for each candidate regularization parameter
sure_reg = [sure(sy, reglist[ir], sig0^2, M, N) for ir=1:Nr]
reg_best = reglist[argmin(sure_reg)] # SURE pick for β


# ### Plot NRMSE and NRMSD versus regularization parameter
plot(xtick=[round(reg_best,digits=3), 6], ytick=0:20:100, legend=:bottomright,
    xlabel=L"\beta", ylabel="'Error' [%]", xlim=(0,6), ylim=(0, 100))
scatter!(reglist, nrmse_reg, color=:blue,
    label=L"\mathrm{NRMSE\ } \|\hat{X}_\beta - X\|_F / \|X\|_F \cdot 100\%",)
scatter!(reglist, nrmsd_reg, color=:red,
    label=L"\mathrm{NRMSD\ } \|\hat{X}_\beta - Y\|_F / \|Y\|_F \cdot 100\%")
scatter!(reglist, sqrt.(sure_reg)/norm(Y)*100, color=:green,
    label=L"(\mathrm{SURE}(\beta))^{1/2} / \|Y\|_F \cdot 100\%")

#
prompt()

#src savefig("06_lr_sure1c.pdf")


# ### Examine shrunk singular values for best regularization parameter
sh = soft.(sy,reg_best)
plot(xtick=[1, Ktrue, sum(sh .!= 0), N], ytick=0:6,
    xlabel=L"k", ylabel=L"\sigma", xlim=(1,N), ylim=(0,5.5))
scatter!(1:N, sy, color=:blue, label=L"\sigma_k(Y) \ \mathrm{noisy}")
scatter!(1:N, sx, color=:red, label=L"\sigma_k(X) \ \mathrm{noiseless}")
scatter!(1:N, sh, color=:green, label=L"\hat{\sigma}_k \ \mathrm{SURE} \ \hat{\beta}")

#
prompt()

#src savefig("06_lr_sure1t.pdf")


# ## Reproducibility

# This page was generated with the following version of Julia:

io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')


# And with the following package versions

import Pkg; Pkg.status()
