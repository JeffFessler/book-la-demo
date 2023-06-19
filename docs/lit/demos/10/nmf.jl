#=
# [Non-negative matrix factorization](@id nmf1)

Non-negative matrix factorization
of hand-written digits
in Julia.
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
        "LinearAlgebra"
        "MIRTjim"
        "MLDatasets"
        "NMF"
        "Plots"
    ])
end


# Tell this Julia session to use the following packages.
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


# Look at image data
pd = jim(data[:,:,1:50,:], "Data";
    colorbar=nothing, size=(600,400), tickfontsize=6, ncol=25)

## savefig(pd, "nmf-data.pdf")

Y = reshape(data, nx*ny, :) # unfold
K = 20
out = nnmf(Y, K)

W = reshape(out.W, nx, ny, K)
## H = out.H
pw = jim(W/maximum(W); title="Factor images", color=:cividis)

U = reshape(svd(Y).U[:,1:K], nx, ny, K)
pu = jim(U/maximum(U); title="Left singular vectors", color=:cividis)

#
#prompt()

#todo include("../../../inc/reproduce.jl")
