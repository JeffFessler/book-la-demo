#=
# [Sparse spectral clustering (SSC)](@id sparse-spectral-cluster)

This example illustrates
[sparse spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering)
using POGM
applied to simulated data
and (todo) hand-written digits.

Original version by Javier Salazar Cavazos.
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
        "Clustering"
        "Distributions"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRT"
        "MIRTjim"
        "MLDatasets"
        "Plots"
        "Random"
    ])
end

# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using Clustering: kmeans
using Distributions: Normal, Random
using InteractiveUtils: versioninfo
using LinearAlgebra: Diagonal, eigen, I, opnorm
using MIRT: pogm_restart
using MIRTjim: jim, prompt
using Plots: default, gui, palette, plot, plot!, scatter, scatter!
using Random: seed!
default(); default(markersize=5, markerstrokecolor=:auto, label="")

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Synthetic data
todo explain
=#

Random.seed!(3) # fix random generation for better debugging

d = Normal(0, 0.5) # noise distributions for x and y locations
e = Normal(0, 0.5)
xval = -9.5:1:9.5 # x locations before adding noise

x1 = xval .+ rand(e, 20) # noisy data that lies on union of subspaces
x2 = xval .+ rand(e, 20)
y1 = 1 * x1 .+ rand(d, 20) # y=x and y=-x is the subspace
y2 = -1 * x2 .+ rand(d, 20)

data = [ [x1';y1'] [x2';y2'] ] # gathering data to one matrix
clusters = [1*ones(20,1); 2*ones(20,1)]; # clusters of our data to compare method

#=
permuteOrder = randperm(40)[1:40] # permute our points in data matrix to be safe
data = data[:,permuteOrder]
clusters = clusters[permuteOrder]
=#

plot(aspect_ratio = 1, size = (550, 500))
plot!(xval, 1 .* xval) # plot subspaces and data points
plot!(xval, -1 .* xval)
scatter!(x1, y1, color=1)
scatter!(x2, y2, color=2)


#=
## POGM for SSC
todo explain
=#

z_lam = 0.001 # regularization parameter for sparsity term
Lf = opnorm(data,2)^2 # Lipschitz constant for ∇f: smooth term of obj function
npoint = size(data,2) # total # of points
x0 = zeros(npoint, npoint) # initialize solution
M = 1 .- Diagonal(ones(npoint)) # mask to force diag(C)=0

# smooth grad is -Y'(Y-YC) but must force diag elements to zero for each step
grad = x -> M .* (-1*data' * (data - data*x)) # gradient of smooth term
soft = (x,t) -> sign(x) * max(abs(x) - t, 0) # soft threshold at t
g_prox = (z,c) -> soft.(z, c * z_lam) # proximal operator for c*b*|x|_1

niter = 1000
A, _ = pogm_restart(x0, x->0, grad, Lf ; g_prox, niter) # POGM method
jim(A, "A")


#=
## Spectral clustering
todo explain
=#

W = transpose(abs.(A)) + abs.(A) # Weight matrix, force Hermitian
jim(W, "W")

D = vec(sum(W, dims=2)) # degree matrix of graph
D = D .^ (-1/2)
D = Diagonal(D) # normalized symmetric Laplacian formula
L = I - D * W * D
jim(L, "L")

E = eigen(L) # eigen value decomposition, really only need vectors
# eigenVectors = eigvecs(L)
# for K=2 subspaces we pick the bottom K eigenvectors (smallest λ)
eigenVectors = E.vectors[:, 1:2]

K = 2
# subspaces so we look for K clusters in rows of eigenvectors
results = kmeans(eigenVectors', K)
assign = results.assignments; # store assignments

seriescolor = palette([:orange, :skyblue], 2)
p4 = scatter(eigenVectors[:,1], eigenVectors[:,2],
 title="Spectral Embedding Plot",
 marker_z = clusters;
# legend=false,
 seriescolor)

# plot truth on the left
p1 = scatter(data[1,:], data[2,:];
 aspect_ratio = 1, size = (550, 450),
 xlims = (-11,11), ylims = (-11,11),
 marker_z = clusters,
 seriescolor,
 title = "Truth",
)
# plot ssc results on the right
p2 = scatter(data[1,:], data[2,:];
 aspect_ratio = 1, size = (550, 450),
 xlims = (-11,11), ylims = (-11,11),
 marker_z = assign,
 seriescolor, 
 title = "SSC (POGM)",
)
p12 = plot(p1, p2, layout = (1, 2), size=(1100, 450))

include("../../../inc/reproduce.jl")
