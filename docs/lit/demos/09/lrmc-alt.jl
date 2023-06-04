#=
# [Low-rank matrix completion via alternating projection](@id lrmc-alt)

This example illustrates
low-rank matrix completion
via alternating projection
using the Julia language.

(This approach is related to "projection onto convex sets" (POCS) methods,  
but the term "POCS" would be a misnomer here
because the rank constraint is not a convex set.)

History:
* 2021-08-23 Julia 1.6.2  
* 2021-12-09 Julia 1.6.4 and use M not Ω
* 2023-06-04 Julia 1.9.0 in Literate

=#

#srcURL

#=
## Setup
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
    ])
end


# Tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LinearAlgebra: svd, svdvals, rank, norm, Diagonal
using LaTeXStrings
using MIRTjim: jim, prompt
using Plots: default, gui, plot, savefig, scatter!, xlabel!
using Plots.PlotMeasures: px
using Random: seed!
using Statistics: mean
default(markersize=7, markerstrokecolor=:auto, label = "",
 tickfontsize = 10, legendfontsize = 18,
)

# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Latent matrix
make a matrix that has low rank
=#
tmp = [
    zeros(1,20);
    0 1 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 1 1 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 1 0 0 1 0 0;
    0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0;
    0 0 1 1 1 1 0 0 0 0 1 1 0 0 0 0 0 1 1 0;
    zeros(1,20)
]';
rank(tmp)

Xtrue = kron(10 .+ 80*tmp, ones(9,9))
rank(Xtrue)

# consistent size
jim1 = (X ; kwargs...) -> jim(X; size = (600,300),
 leftmargin = 10px, rightmargin = 10px, kwargs...)
# consistent display range
jimc = (X ; kwargs...) -> jim1(X; clim=(0,100), kwargs...)
pt = jimc(Xtrue)
## savefig(pt, "mc_ap_x.pdf")

#=
## Noisy / incomplete data 
=#
seed!(0)
M = rand(Float32, size(Xtrue)) .>= 0.75 # 75% missing
Y = M .* (Xtrue + randn(size(Xtrue)));

py = jimc(Y ; title="Y: Corrupted image matrix\n(missing pixels set to 0)")
nrmse = (Xh) -> round(norm(Xh - Xtrue) / norm(Xtrue) * 100, digits=1)
xnrmse! = (Xh) -> xlabel!("NRMSE = $(nrmse(Xh)) %")
@show nrmse(Y)
xnrmse!(Y)
## savefig(py, "mc_ap_y.pdf")

#=
### What is rank(Y) ??
* A 5-9  
* B 10-49  
* C 50-59  
* D 60-70  
* E 71-200  
=#

#src rank(Y)
#src svdvals(Y)

# Show mask, count proportion of missing entries
frac_nonzero = count(M) / length(M)
pm = jim1(M; title="M : Locations of observed entries",
    xlabel = "sampled fraction = $(round(frac_nonzero * 100, digits=1))%")
## savefig(pm, "mc_ap_m.pdf")

#=
## Low-rank approximation
A simple low-rank approximation works poorly for missing data.
=#
r = 5
U,s,V = svd(Y)
Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]'
pr = jimc(Xr ; title="low-rank approximation of data Y, r=$r")
xnrmse!(Xr)
## savefig(pr, "mc_ap_lr.pdf")


#=
## Alternating projection
Alternating projection is an
iterative method that alternates
between projecting onto the set of rank-5 matrices
and onto the set of matrices that match the data.
=#

niter = 400
r = 5
function lrmc_alt(Y)
    Xr = copy(Y)
    Xr[.!M] .= mean(Y[M]) # fill missing values with mean of other values
    @show nrmse(Xr)
    for iter in 1:niter
        U,s,V = svd(Xr)
        Xr = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]' # project onto "𝒞" &Cscr; U+1D49E
        Xr[M] .= Y[M] # project onto "𝒟" &Dscr; U+1D49F
        if 0 == iter % 40
            @show nrmse(Xr)
        end
    end
    return Xr
end
Xr = lrmc_alt(Y)
pa = jimc(Xr ; title="'Alternating Projection' result at $niter iterations")
xnrmse!(Xr)
## savefig(pa, "mc_ap_400.pdf")


#=
### What is rank(Xr) here ??
* A 5-9  
* B 10-49  
* C 50-59  
* D 60-70  
* E 71-200  
=#

## rank(Xr)
## svdvals(Xr)

# Run one more projection step onto the set of rank-r matrices
U,s,V = svd(Xr)
Xfinal = U[:,1:r] * Diagonal(s[1:r]) * V[:,1:r]';

pf = jimc(Xfinal ; title="'Alternating Projection' result at $niter iterations")
xnrmse!(Xfinal)
## savefig(pf, "mc_ap_xh.pdf")

#=
### What is rank(Xfinal) here ??
* A 5-9  
* B 10-49  
* C 50-59  
* D 60-70  
* E 71-200  
=#

## rank(Xfinal)

# Plot singular values
sp = svdvals(Xr)
effective_rank = sum(sp .> (0.01*sp[1]))

#
ps = plot(title="singular values",
 xaxis=(L"k", (1, minimum(size(Y))), [1, effective_rank, minimum(size(Y))]),
 yaxis=(L"σ",), labelfontsize = 18,
 leftmargin = 25px, bottommargin = 20px, size = (600,350), widen = true,
)
scatter!(svdvals(Y), color=:red, label="Y (data)")
scatter!(svdvals(Xtrue), color=:blue, label="Xtrue")
scatter!(sp, color=:green, label="Alt. Proj. output")

## savefig(ps, "mc_ap_sv.pdf")

#
prompt()

include("../../../inc/reproduce.jl")
