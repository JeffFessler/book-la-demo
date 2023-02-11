#=
# [spectral-cluster](@id spectral-cluster)

## Spectral clustering illustration

This example illustrates spectral clustering
applied to hand-written digits.

This page was generated from a single Julia file:
[spectral-cluster.jl](@__REPO_ROOT_URL__/spectral-cluster.jl).
=#

#md # In any such Julia documentation,
#md # you can access the source code
#md # using the "Edit on GitHub" link in the top right.

#md # The corresponding notebook can be viewed in
#md # [nbviewer](https://nbviewer.org/) here:
#md # [`spectral-cluster.ipynb`](@__NBVIEWER_ROOT_URL__/spectral-cluster.ipynb),
#md # and opened in [binder](https://mybinder.org/) here:
#md # [`spectral-cluster.ipynb`](@__BINDER_ROOT_URL__/spectral-cluster.ipynb).


# ### Setup

# Packages needed here.

using LinearAlgebra: I, norm, Diagonal, eigen
using StatsBase: mean
using MLDatasets: MNIST
using Random: seed!, randperm
using LaTeXStrings # pretty plot labels
using Plots: default, gui, plot, scatter, plot!, scatter!
using MIRTjim: jim, prompt
using InteractiveUtils: versioninfo
using Clustering: kmeans
default(markersize=5, markerstrokecolor=:auto, label="")

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
    digitn = (0, 1, 3) # which digits to use
    isinteractive() || (ENV["DATADEPS_ALWAYS_ACCEPT"] = true) # avoid prompt
    dataset = MNIST(Float32, :train)
    nrep = 30
    ## function to extract the 1st 1000 examples of digit n:
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


# Look at "unlabeled" image data for unsupervised clustering
jim(data)
#src savefig("spectral-cluster-data.pdf")

# Choose similarity function
σ = 2^-2 # tuning parameter
sfun(x,z) = exp(-norm(x-z)^2/nx/ny/σ^2)

# Weight matrix
slices = eachslice(data, dims=3)
W = [sfun(x,z) for x in slices, z in slices]
pw = jim(W, "weight matrix W")

# Degree matrix
D = Diagonal(vec(sum(W; dims=2)))

# Normalized graph Laplacian
L = I - inv(D) * W
jim(L, "Normalized graph Laplacian L")

# Eigendecomposition and eigenvalues
eig = eigen(L)
pe = scatter(eig.values, xlabel = L"k", ylabel="Eigenvalues")

#
prompt()

# Apply k-means++ to eigenvectors
K = length(digitn) # cheat: using the known number of digits
Y = eig.vectors[:,1:K]'
rc = kmeans(Y, K)

# Confusion matrix using class assignments from kmeans++
label_list = unique(labels)
#src assign_list = unique(rc.assignments) # 1:K

result = zeros(Int, K, length(label_list))
for k in 1:K # each cluster
    rck = rc.assignments .== k
    for (j,l) in enumerate(label_list)
        result[k,j] = count(rck .& (l .== labels))
    end
end
result

# Visualize the clustered digits
pc = jim(
 [jim(data[:,:,rc.assignments .== k], "Class $k"; prompt=false) for k in 1:K]...
)

#=
The clustering here seems only so-so,
at least from the digit classification point of view.
Each of these digits lives reasonably close
to a manifold,
and apparently the simply Gaussian similarity function
used here does not adequately capture
within-manifold similarities.
=#


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
