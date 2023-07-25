#=
# [Binary classification](@id class01)

Binary classification of hand-written digits
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
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "MLDatasets"
        "Plots"
        "Random"
        "StatsBase"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings # nice plot labels
using LinearAlgebra: dot
using MIRTjim: jim, prompt
using MLDatasets: MNIST
using Plots: default, gui, savefig
using Plots: histogram, histogram!, plot
using Plots: RGB, cgrad
using Plots.PlotMeasures: px
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
    digitn = (0, 1) # which digits to use
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
    data = reshape(data, nx, ny, :)
    seed!(0)
    tmp = randperm(nrep * ndigit)
    data = data[:,:,tmp]
    labels = labels[tmp]
    size(data) # (nx, ny, nrep*ndigit)
end


# Look at "unlabeled" image data
pd = jim(data, "Data"; size=(600,300), tickfontsize=8,)

# Extract training data
data0 = data[:,:,labels .== 0]
data1 = data[:,:,labels .== 1];

pd0 = jim(data0[:,:,1:36]; nrow=6, colorbar=nothing, size=(400,400))
pd1 = jim(data1[:,:,1:36]; nrow=6, colorbar=nothing, size=(400,400))
## savefig(pd0, "class01-0.pdf")
## savefig(pd1, "class01-1.pdf")

# red-black-blue colorbar:
RGB255(args...) = RGB((args ./ 255)...)
color = cgrad([RGB255(230, 80, 65), :black, RGB255(23, 120, 232)]);

#=
## Weights
Compute sample average of each training class
and define classifier weights as differences of the means.
=#
μ0 = mean(data0, dims=3)
μ1 = mean(data1, dims=3)
w = μ1 - μ0; # hand-crafted weights

# images of means and weights
siz = (540,400)
args = (xaxis = false, yaxis = false) # book
p0 = jim(μ0; clim=(0,1), size=siz, cticks=[0,1], args...)
p1 = jim(μ1; clim=(0,1), size=siz, cticks=[0,1], args...)
pw = jim(w; color, clim=(-1,1).*0.8, size=siz, cticks=(-1:1)*0.75, args...)
#src jim(w; color=:cividis)
pm = plot( p0, p1, pw;
  size = (1400, 350),
  layout = (1,3),
  rightmargin = 20px,
)
## savefig(p0, "class01-0.pdf")
## savefig(p1, "class01-1.pdf")
## savefig(pw, "class01-w.pdf")
## savefig(pm, "class01-mean.pdf")

#=
## Inner products
Examine performance of simple linear classifier.
(Should be done with test data, not training data...)
=#
i0 = [dot(w, x) for x in eachslice(data0, dims=3)]
i1 = [dot(w, x) for x in eachslice(data1, dims=3)];

bins = -80:20
ph = plot(
 xaxis = (L"⟨\mathbf{\mathit{v}},\mathbf{\mathit{x}}⟩", (-80, 20), -80:20:20),
 yaxis = ("", (0, 25), 0:10:20),
 size = (600, 250), bottommargin = 20px,
)
histogram!(i0; bins, color=:red , label="0")
histogram!(i1; bins, color=:blue, label="1")

## savefig(ph, "class01-hist.pdf")

#
prompt()

include("../../../inc/reproduce.jl")
