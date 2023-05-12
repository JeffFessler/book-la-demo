#=
# [Laplacian eigenmaps](@id eigmap)

This example illustrates
Laplacian eigenmaps
using the Julia language.
These eigenmaps
provide nonlinear dimensionality reduction
for data lying near manifolds
(rather than near subspaces).

See
[Belkin & Niyogi, 2003](https://doi.org/10.1162/089976603321780317).
=#

#srcURL

#=
Add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "ImagePhantoms"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ImagePhantoms: rect, phantom
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: norm, Diagonal, eigen, svd, svdvals
using MIRTjim: jim, prompt
using Plots: plot, scatter, savefig, default
using Random: seed!
default(); default(label = "", markerstrokecolor = :auto)
seed!(0)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Generate data

The example considered here
uses synthetic data
consisting of images of rectangles
of various widths and rotations.

The only latent parameters here
are one of the rectangle widths
and the rotation angle.
So all of the images
lie in a manifold of dimension 2
in a ``16^2 = 256`` dimensional
ambient space.

=#

function rect_phantom( ;
    T = Float32,
    nx = 40,
    ny = 40,
    sigma1::Real = 0f0,
    sigma2::Real = 0f0,
    level1::Function = () -> 0 + sigma1 * randn(T),
    level2::Function = () -> 1 + sigma2 * randn(T),
    angle::Function = () -> rand(T) * π/4f0,
    width_min::Real = min(nx,ny) / 4f0 + 3,
    width_max::Real = min(nx,ny) / 1.6f0, # √2
    width::Function = () -> [width_min/2,
        width_min + rand(T) * (width_max - width_min)],
    center::Function = () -> 0 * (rand(T, 2) .- 1//2),
    oversample::Int = 6,
)
    ob = rect(Tuple(center()), Tuple(width()), angle())
    l1 = level1()
    l2 = level2()
    x = (1:nx) .- (nx-1)/2
    y = (1:ny) .- (ny-1)/2
    return l1 .+ (l2 - l1) * phantom(x, y, [ob], oversample),
        ob.angle[1], ob.width[2]
end

function make_phantoms(nx, ny, nrep)
    data = zeros(Float32, nx, ny, nrep)
    angles = zeros(nrep)
    widths = zeros(nrep)
    for iz in 1:nrep
        data[:,:,iz], angles[iz], widths[iz] = rect_phantom(; nx, ny)
    end
    return data, angles, widths
end


if !@isdefined(data)
    nx,ny = 40,40
    nrep = 500
    @time data, angles, widths = make_phantoms(nx, ny, nrep)
    pj = jim(data[:,:,1:88]; title = "88 of $nrep images")
end
#src savefig(pj, "eigmap-data.pdf")


#=
These data do not lie in a 2-dimensional subspace.
To see that,
we plot the first 40 singular values.
Even in the absence of noise here,
there are many singular values
that are are far from zero.
=#
tmp = svdvals(reshape(data, nx*ny, nrep))
ps = scatter(tmp; title = "Data singular values", widen=true,
 xaxis = (L"k", (1, 40), [1, 40, nx*ny]),
 yaxis = (L"σ_k", (0, 216), [0, 32, 48, 72, 215]),
)
#src savefig(ps, "eigmap-svd.pdf")

#
prompt()


#=
## Eigenmaps

Apply one of the Laplacian eigenmap methods.
First compute the distances between all pairs of data points (images).

There is little if any humanly visible structure
in the distance map.
=#

distance = [norm(d1 - d2) for
    d1 in eachslice(data; dims=3),
    d2 in eachslice(data; dims=3)
]
pd = jim(distance; title = L"‖ X_j - X_i \; ‖", xlabel = L"i", ylabel = L"j")
#src savefig(pd, "eigmap-dis.pdf")


#=
Compute the weight matrix
that describes an affinity
between data points ``i`` and ``j``.
There are many ways to do this;
here we follow the approach given in
[Sanders et al. 2016](https://doi.org/10.1109/tmi.2016.2576899).
=#

α = Float64(sum(abs2, distance) / nrep^2) # per Sanders eqn. (4)
W = @. exp(-distance^2 / α)
pw = jim(W; title = L"W_{ij}", xlabel = L"i", ylabel = L"j")
#src savefig(pw, "eigmap-w.pdf")


#=
## Graph Laplacian

Compute the
[graph Laplacian](https://en.wikipedia.org/wiki/Laplacian_matrix)
from the weight matrix.
=#

d = vec(sum(W; dims=2))
D = Diagonal(d)
L = D - W # Laplacian
pl = jim(L; title = L"L_{ij}", xlabel = L"i", ylabel = L"j", color=:cividis)
#src savefig(pl, "eigmap-l.pdf")

#=
Compute the
[generalized eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Generalized_eigenvalue_problem)
to find solutions of
``L v = λ D v``.

Because the vector ``\mathbf{1}``
is in the null space of ``L``,
the first (smallest) generalized eigenvalue is 0.
=#
F = eigen(L, Matrix(D))

pe = scatter(F.values;
 title = "Generalized eigenvalues",
 xlabel=L"k", ylabel=L"λ_k",
 xticks = [1,nrep],
 yticks = [0, 0.7, 0.9, 1],
)
#src savefig(pe, "eigmap-eigval.pdf")

#
prompt()

#=
Here we are interested in the first couple
of generalized eigenvectors,
excluding the one corresponding to ``λ=0``.
Those vectors serve as features
for the reduced dimension.
=#
features = 1000 * F.vectors[:,2:3]
pf = scatter(features[:,1], features[:,2];
  title = "Eigenmap features",
  xlabel="feature 1", ylabel="feature 2")
#src savefig(pf, "eigmap-f.pdf")

#
prompt()

#=
To examine whether those features
encode the relevant manifold here,
we plot the features
with color-coding
corresponding to the rectangle
width and angle.

The first feature
is correlated with
rectangle angle.
The second feature
is correlated with
rectangle width.
=#

pc = plot(
 scatter(features[:,1], features[:,2], marker_z = widths, color=:cividis,
  xlabel="feature 1", ylabel="feature 2",
  colorbar_title="width",
 ),
 scatter(features[:,1], features[:,2], marker_z = angles, color=:cividis,
  xlabel="feature 1", ylabel="feature 2",
  colorbar_title="angle",
 ),
)
#src savefig(pc, "eigmap-c.pdf")

#
prompt()

#=
As another way to illustrate this correlation,
suppose we examine all images
where feature 1 is in some interval like ``(-1.5,-1)``.
These images are all of rectangles
of similar orientation, but various widths.
=#

tmp = data[:,:, -1.5 .< features[:,1] .< -1]
pf1 = jim(tmp; title = "Feature 1 set")
#src savefig(pf1, "eigmap-pf1.pdf")


#=
Conversely,
the images
where feature 2 is in some interval like ``(0.2,0.5)``
are all rectangles
of similar widths, but various rotations.
=#

tmp = data[:,:, 0.2 .< features[:,2] .< 0.5]
pf2 = jim(tmp; nrow=2, title = "Feature 2 set", size=(600,300))
#src savefig(pf2, "eigmap-pf2.pdf")


#src X = [features ones(nrep)]
#src scatter(widths, X * (X \ widths))
#src scatter(angles, X * (X \ angles))
#src Y = [widths angles ones(nrep)]
#src scatter(features[:,1], Y * (Y \ features[:,1]))
#src scatter(features[:,2], Y * (Y \ features[:,2]))


# Examine PCA for comparison

tmp = reshape(data, nx*ny, nrep)
U2 = svd(tmp).U[:,1:2]
pup = jim(reshape(U2, nx, ny, 2); title="First 2 PCA components")
#src savefig(pup, "eigmap-pca-u.pdf")

# Projection onto a 2-dimensional subspace works poorly:
lr2 = reshape(U2 * (U2' * tmp), nx, ny, :)
plr = jim(lr2[:,:,1:88], "Projection onto 2-dimensional subspace")
#src savefig(plr, "eigmap-pca-lr2.pdf")

# Nevertheless, the corresponding features
# are reasonably correlated with width and angle.
feat_pca = (U2' * tmp)' # leading coefficients

pp = plot(
 scatter(feat_pca[:,1], feat_pca[:,2], marker_z = widths, color=:cividis,
  xlabel="feature 1", ylabel="feature 2",
  colorbar_title="width",
 ),
 scatter(feat_pca[:,1], feat_pca[:,2], marker_z = angles, color=:cividis,
  xlabel="feature 1", ylabel="feature 2",
  colorbar_title="angle",
 ),
 plot_title = "First two PCA coefficients",
)
#src savefig(pp, "eigmap-pca-f.pdf")

#
prompt()

include("../../../inc/reproduce.jl")
