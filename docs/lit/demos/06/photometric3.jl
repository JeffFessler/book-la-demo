#=
# [Photometric Stereo](@id photometric3)

This example illustrates
todo
using the Julia language.
=#

#srcURL

#=
First we add the Julia packages that are need for this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "InteractiveUtils"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using Downloads: download
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, svd, rank
using MIRTjim: jim, prompt
using NPZ: npzread
using Plots; default(label="", markerstrokecolor=:auto)
using Printf: @printf
using Random: seed!


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Generate data

# Noisy data with slope=1.  Both x and y values are noisy!
seed!(0)

#=
Data from
[2012 CVPR paper by Ikehata et al.](https://doi.org/10.1109/CVPR.2012.6247691)
=#

if !@isdefined(images)
    url0 = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/bunny_lambert/image000.npy"

    tmp = download(url0)
    x = npzread(tmp)

    dim = size(x)[1:2]
    nimage = 50
    images = zeros(Float32, dim..., nimage)
    images[:,:,1] = x[:,:,1]'

    for id in 2:nimage
        id3 = @sprintf("%03d", id-1)
    #   @show id3
        url = replace(url0, "000" => id3)
        @show url
        xtmp = npzread( download(url) )
        images[:,:,id] = xtmp[:,:,1]'
    end

    i1 = 22:256-25
    i2 = 28:217
    images = images[i1,i2,1:2:end]
    dim = size(images)
## https://raw.githubusercontent.com/yasumat/RobustPhotometricStereo/master/data/bunny/mask.png
end

jim(images)

# ## Rank-3 approximation

# To make a low-rank approximation, collect data into a matrix
X = reshape(images, :, dim[3])

# Examine singular values
U, s, V = svd(X)
s # 1st 3 singular values are larger than rest
scatter(s, xlabel=L"k", ylabel=L"σ_k")


# Construct rank-3 approximation
B = U[:,1:3] * Diagonal(s[1:3]) * V[:,1:3]' # rank-3 approximation
rank(B)

tmp = reshape(U[:,1:3], dim[1:2]..., 3)
jim(tmp)

# todo: depth

#
prompt()
throw()

#
B


# ### Plot rank-1 approximation
xb = B[1,:]
yb = B[2,:]

lineplot(pl, (xb\yb)[1], :black, "")
scatter!(pl, xb, yb, color=:black, markersize=5, marker=:square, label="rank1")

#
prompt()


# ## Use least-squares estimation to estimate slope:
slope = y'*x / (x'*x) # cf inv(A'A) * A'b
slope = (x \ y)[1] # cf A \ b


# ### Plot the LS fit and the low-rank approximation on same graph
lineplot(pl, slope, :green, "LS")

#
prompt()

#src savefig("06_low_rank1_all.pdf")


# ## Illustrate the Frobenius norm approximation error graphically
pl = plotdata()
for i in 1:length(xb)
    plot!(pl, [x[i], xb[i]], [y[i], yb[i]], color=:black, width=2)
end
lineplot(pl, (xb\yb)[1], :black, "")
scatter!(pl, xb, yb, color=:black, markersize=5, marker=:square, label="rank1")

#
prompt()

#src savefig("06_low_rank1_r1.pdf")


# ## Illustrate the LS residual graphically
xl = x; yl = slope*xl # LS points
pl = plotdata()
for i in 1:length(x)
    plot!(pl, [x[i], xl[i]], [y[i], yl[i]], color=:green, width=2)
end
lineplot(pl, slope, :green, "")
scatter!(pl, xl, yl, color=:green, markersize=5, marker=:square, label="LS")

#
prompt()

#src savefig("06_low_rank1_ls.pdf")


include("../../../inc/reproduce.jl")
