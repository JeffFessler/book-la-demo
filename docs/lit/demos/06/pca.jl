#=
# [PCA](@id pca)

## Principal component analysis (PCA) illustration

This example illustrates
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)
of hand-written digit data.
=#

#srcURL

# ### Setup

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
        "MLDatasets"
        "Plots"
        "Random"
        "StatsBase"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings # nice plot labels
using LinearAlgebra: svd
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Plots: default, gui, plot, savefig, scatter, scatter!
using Random: seed!, randperm
using StatsBase: mean
default(); default(markersize=5, markerstrokecolor=:auto, label="",
 tickfontsize=14, labelfontsize=18, legendfontsize=18, titlefontsize=18)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Load data

Read the MNIST data for some handwritten digits.
This code will automatically download the data from web if needed
and put it in a folder like: `~/.julia/datadeps/MNIST/`.
=#
if !@isdefined(data)
    digitn = (0, 1, 4) # which digits to use
    isinteractive() || (ENV["DATADEPS_ALWAYS_ACCEPT"] = true) # avoid prompt
    dataset = MNIST(Float32, :train)
    nrep = 60 # how many of each digit
    ## function to extract the 1st `nrep` examples of digit n:
    data = n -> dataset.features[:,:,findall(==(n), dataset.targets)[1:nrep]]
    data = cat(dims=4, data.(digitn)...)
    labels = vcat([fill(d, nrep) for d in digitn]...) # to check later
    nx, ny, nrep, ndigit = size(data)
    data = data[:,2:ny,:,:] # make images non-square to force debug
    ny = size(data,2)
    data = reshape(data, nx, ny, :)
    seed!(0)
    tmp = randperm(nrep * ndigit)
    data = data[:,:,tmp]
    labels = labels[tmp]
    size(data) # (nx, ny, nrep*ndigit)
end


# Look at "unlabeled" image data prior to unsupervised dimensionality reduction
pd = jim(data, "Data"; size=(600,300), tickfontsize=8,)
#src savefig(pd, "pca-data.pdf")

# Compute sample average of data
μ = mean(data, dims=3)
pm = jim(μ, "Mean")
#src savefig(pm, "pca-mean.pdf")

# Scree plot of singular values
data2 = reshape(data .- μ, :, nrep*ndigit) # (nx*ny, nrep*ndigit)
f = svd(data2)
ps = scatter(f.S; title="Scree plot", widen=true,
 xaxis = (L"k", (1,ndigit*nrep), [1, 6, ndigit*nrep]),
 yaxis = (L"σ_k", (0,48), [0, 0, 47]),
)
#src savefig(ps, "pca-scree.pdf")

#
prompt()

#=
The first 6 or so singular values are notably larger than the rest,
but for simplicity of visualization here
we just use the first two components.
=#
K = 2
Q = f.U[:,1:K]
pq = jim(reshape(Q, nx,ny,:), "First $K singular components"; size=(600,300))
#src savefig(pq, "pca-q.pdf")

#=
Now use the learned subspace basis `Q`
to perform dimensionality reduction.
The resulting coefficients are called
"factors" in
[factor analysis](https://en.wikipedia.org/wiki/Factor_analysis)
and
"scores" in
[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis).
=#
z = Q' * data2 # (K, nrep*ndigit)

#=
Examine the PCA scores.
The three digits are remarkably well separated
even in just two dimensions.
=#
pz = plot(title = "Score plot for $ndigit digits",
 xaxis=("Score 1", (-5,8), -3:3:6),
 yaxis=("Score 2", (-6,4), -4:4:4),
) 
for d in digitn
    scatter!(z[1,labels .== d], z[2,labels .== d], label="Digit $d")
end
#src savefig(pz, "pca-score.pdf")

#
prompt()

include("../../../inc/reproduce.jl")
