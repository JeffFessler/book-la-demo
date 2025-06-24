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
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, eigen, I, opnorm
using MIRT: pogm_restart
using MIRTjim: jim, prompt
using Plots: default, gui, palette, plot, plot!, scatter, scatter!
using Random: randperm, seed!
default(); default(markersize=5, markerstrokecolor=:auto, label="")

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Synthetic data

Generate synthetic data points in ℝ²
that lie along `K = 2` subspaces
in the span of `(1,1)` and `(1,-1)`.
=#

seed!(3) # fix random generation for better debugging

xval = -9.5:1:9.5 # x locations before adding noise

N = length(xval)
σ = 0.5
x1 = xval + σ * randn(N) # noisy data that lies on union of subspaces
x2 = xval + σ * randn(N)
y1 = 1 * x1 .+ σ * randn(N) # y=x and y=-x are the 2 subspaces
y2 = -1 * x2 .+ σ * randn(N)

data = [ [x1';y1'] [x2';y2'] ] # gathering data to one matrix
clusters = [1*ones(20,1); 2*ones(20,1)]; # ground-truth clusters

if true # permute data points
    permuteOrder = randperm(40)
    data = data[:,permuteOrder]
    clusters = clusters[permuteOrder]
end
reord = invperm(permuteOrder)

# plot subspaces and data points
p0 = plot(aspect_ratio = 1, size = (550, 500), xlabel=L"x_1", ylabel=L"x_2")
plot!(p0, xval, 1 .* xval)
plot!(p0, xval, -1 .* xval)
pd = deepcopy(p0)
scatter!(pd, x1, y1, color=1)
scatter!(pd, x2, y2, color=2)
plot!(pd, title = "Data and Subspaces")


#=
## POGM for SSC

Solve the SSC problem with the self-representation cost function
```
\arg\min_C (1/2) ‖ Y (M ⊙ C) - Y ‖_{\mathrm{F}}² + λ ‖ C ‖_{1,1}
```
where `M` is a mask matrix
that is unity everywhere except 0 along the diagonal
that forces each column of `Y`
to be represented as a (sparse) linear combination
of *other* columns of `Y`.
The regularizer encourages sparsity of `C`.

POGM is an optimal accelerated optimization method
for convex composite cost functions.
- [https://doi.org/10.1007/s10957-018-1287-4](https://doi.org/10.1007/s10957-018-1287-4)
- [https://doi.org/10.1137/16m108104x](https://doi.org/10.1137/16m108104x)
=#

λ = 0.001 # regularization parameter for sparsity term
Lf = opnorm(data,2)^2 # Lipschitz constant for ∇f: smooth term of obj function
npoint = size(data,2) # total # of points
x0 = zeros(npoint, npoint) # initialize solution
M = 1 .- I(npoint); # mask to force diag(C)=0

grad(x) = M .* (data' * (data*x - data)) # ∇f: gradient of smooth term
soft(x,t) = sign(x) * max(abs(x) - t, 0) # soft threshold at t
g_prox(z,c) = soft.(z, c * λ) # proximal operator for c*b*|x|_1

niter = 1000
A, _ = pogm_restart(x0, x->0, grad, Lf ; g_prox, niter) # POGM method
jim(A[reord,reord], "A")


#=
## Spectral clustering

Cluster via a spectral method;
see:
- [https://doi.org/10.1109/JSTSP.2018.2867446](https://doi.org/10.1109/JSTSP.2018.2867446)
- [https://doi.org/10.1109/TPAMI.2013.57](https://doi.org/10.1109/TPAMI.2013.57)
=#

W = transpose(abs.(A)) + abs.(A) # Weight matrix, force Hermitian
jim(W[reord,reord], "W")

#
D = vec(sum(W, dims=2)) # degree matrix of graph
D = D .^ (-1/2)
D = Diagonal(D) # normalized symmetric Laplacian formula
L = I - D * W * D
jim(L[reord,reord], "L")

# for K=2 subspaces we pick the bottom K eigenvectors (smallest λ)
K = 2
E = eigen(L) # eigen value decomposition, really only need vectors
#src eigenVectors = eigvecs(L)
eigenVectors = E.vectors[:, 1:K];

seriescolor = palette([:orange, :skyblue], 2)
p4 = scatter(eigenVectors[:,1], eigenVectors[:,2],
 title="Spectral Embedding Plot",
 marker_z = clusters;
 seriescolor,
)

# K subspaces so we look for K clusters in rows of eigenvectors
results = kmeans(eigenVectors', K)
assign = results.assignments; # store assignments

# plot truth on the left
p1 = deepcopy(p0)
scatter!(p1, data[1,:], data[2,:];
 aspect_ratio = 1, size = (550, 450),
 xlims = (-11,11), ylims = (-11,11),
 marker_z = clusters,
 seriescolor,
 title = "Truth",
)
# plot ssc results on the right
p2 = deepcopy(p0)
scatter!(p2, data[1,:], data[2,:];
 aspect_ratio = 1, size = (550, 450),
 xlims = (-11,11), ylims = (-11,11),
 marker_z = assign,
 seriescolor, 
 title = "SSC (POGM)",
)
p12 = plot(p1, p2, layout = (1, 2), size=(1100, 450))

include("../../../inc/reproduce.jl")
