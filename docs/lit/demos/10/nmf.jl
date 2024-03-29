#=
# [Non-negative matrix factorization](@id nmf1)

[Non-negative matrix factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)
of hand-written digit images
in Julia.
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
        "LinearAlgebra"
        "MIRTjim"
        "MLDatasets"
        "NMF"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LinearAlgebra: svd
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using NMF: nnmf
using Plots: default, gui, savefig
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
    digitn = 0:9 # which digits to use
    isinteractive() || (ENV["DATADEPS_ALWAYS_ACCEPT"] = true) # avoid prompt
    dataset = MNIST(Float32, :train)
    nrep = 100 # how many of each digit
    ## function to extract the 1st `nrep` examples of digit n:
    data = n -> dataset.features[:,:,findall(==(n), dataset.targets)[1:nrep]]
    data = cat(dims=4, data.(digitn)...)
    labels = vcat([fill(d, nrep) for d in digitn]...) # to check later
    nx, ny, nrep, ndigit = size(data)
    data = data[:,2:ny,:,:] # make images non-square to force debug
    ny = size(data,2)
    size(data) # (nx, ny, nrep, ndigit)
end


# Look at some of the image data
pd = jim(data[:,:,1:50,:], "Data, M=$(nx*ny), N=$(nrep*ndigit)";
    colorbar=nothing, size=(600,400), tickfontsize=6, ncol=25)

## savefig(pd, "nmf-data.pdf")


#=
## Run NMF
=#
Y = reshape(data, nx*ny, :) # unfold
K = 20
out = nnmf(Y, K)

#=
## Results
Examine the left factor vectors as images.
=#
W = reshape(out.W, nx, ny, K)
## H = out.H
pw = jim(W/maximum(W); title="Left NMF factor images, K=$K", color=:cividis)

## savefig(pw, "nmf-w.pdf")


#=
## SVD basis
Examine the left singular vectors as images.
=#
U = reshape(svd(Y).U[:,1:K], nx, ny, K)
pu = jim(U/maximum(U); title="Left singular vectors", color=:cividis)

## savefig(pu, "nmf-u.pdf")

include("../../../inc/reproduce.jl")
