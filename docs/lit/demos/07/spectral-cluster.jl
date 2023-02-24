#=
# [Spectral clustering](@id spectral-cluster)

## Spectral clustering illustration

This example illustrates
[spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering)
via normalized graph Laplacian
applied to hand-written digits.
=#

#srcURL

# ### Setup

#=
First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "Clustering"
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


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using Clustering: kmeans
using InteractiveUtils: versioninfo
using LaTeXStrings # pretty plot labels
using LinearAlgebra: I, norm, Diagonal, eigen
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Plots: default, gui, plot, scatter, plot!, scatter!
using Random: seed!, randperm
using StatsBase: mean
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

# ## Normalized graph Laplacian
L = I - inv(D) * W
jim(L, "Normalized graph Laplacian L")

# Eigendecomposition and eigenvalues
eig = eigen(L)
pe = scatter(eig.values, xlabel = L"k", ylabel="Eigenvalues")

#
prompt()

# ## Apply k-means++ to eigenvectors
K = length(digitn) # try using the known number of digits
Y = eig.vectors[:,1:K]'
r3 = kmeans(Y, K)

# Confusion matrix using class assignments from kmeans++
label_list = unique(labels)
#src assign_list = unique(r3.assignments) # 1:K

result = zeros(Int, K, length(label_list))
for k in 1:K # each cluster
    rck = r3.assignments .== k
    for (j,l) in enumerate(label_list)
        result[k,j] = count(rck .& (l .== labels))
    end
end
result

# Visualize the clustered digits
p3 = jim(
 [jim(data[:,:,r3.assignments .== k], "Class $k"; prompt=false) for k in 1:K]...
)

#=
The clustering here seems only so-so,
at least from the digit classification point of view.
Each of these digits lives reasonably close
to a manifold,
and apparently the simply Gaussian similarity function
used here does not adequately capture
within-manifold similarities.

However,
there is no reason to think that it is optimal
to use the same number of classes
as digits.
Let's try again using more classes (larger ``K``).
=#

K = 9
Y = eig.vectors[:,1:K]'
r9 = kmeans(Y, K)
p9 = jim(
 [jim(data[:,:,r9.assignments .== k], "Class $k"; prompt=false) for k in 1:K]...
)


#=
Now there is somewhat more consistency
between images in the same class,
=#

include("../../../inc/reproduce.jl")
