#=
# [Photometric Stereo](@id photometric3)

This example illustrates
[photometric stereo](https://en.wikipedia.org/wiki/Photometric_stereo)
for
[Lambertian surfaces](https://en.wikipedia.org/wiki/Lambertian_reflectance)
using the Julia language.

This method determines the surface normals of an object
from 3 or more pictures of the object
taken under different lighting conditions.
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
using LinearAlgebra: Diagonal, svd, svdvals, rank, norm
using MIRTjim: jim, prompt
using NPZ: npzread
using Plots; default(label="", markerstrokecolor=:auto)
using Printf: @sprintf
using Random: seed!


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

# todo
#isinteractive() && prompt(:prompt);


#=
Load ground truth surface normal vectors of "bunny" data used in
[2012 CVPR paper by Ikehata et al.](https://doi.org/10.1109/CVPR.2012.6247691).
=#
if !@isdefined(gt_normal)
    url = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/gt_normal.npy"
    tmp = npzread(download(url))
    i1 = 22:256-25
    i2 = 28:217
    tmp = permutedims(tmp, [2, 1, 3]) # "transpose"
    tmp = tmp[i1, i2, :] # crop
# todo cut:
#   tmp = tmp[:, :, [2, 1, 3]] # transpose?
#   tmp[:,:,3] .*= -1 # sign flip
    gt_normal = tmp
    nx, ny = size(gt_normal)[1:2]
end

function hemi_normal(x, y ; # surface normal of a hemi-ellipsoid
    xh = 70f0,
    yh = 60f0,
    zh = 50f0,
)
    tmp = (x/xh)^2 + (y/yh)^2
    if tmp < 1
        z = zh * sqrt(1 - tmp)
        tmp = [x/xh^2, y/yh^2, z/zh^2]
        return tmp / norm(tmp)

#=
    tmp = (x^2 + y^2) / rhemi^2
    if tmp < 1
        tmp = sqrt(1 - tmp)
        xn = hhemi / tmp / rhemi^2 * (-x)
        yn = hhemi / tmp / rhemi^2 * (-y)
        rn2 = xn^2 + yn^2
#@show rn2
        zn = rn2 < 1 ? sqrt(1 - rn2) : 0
#       zn = sqrt(1 - rn2)
        return [xn, yn, zn]
=#
    end
    return [0, 0, 0]
end

if true
    x = (1:nx) .- (nx+1)/2
    y = (1:ny) .- (ny+1)/2
    tmp = hemi_normal.(x, y')
    tmp = reduce(hcat, tmp)
    gt_normal = reshape(tmp', nx, ny, 3)
end
pn_gt = jim(gt_normal; title="Ground-truth normals", nrow=1,
 xaxis=L"x", yaxis=L"y", size=(600,300))

#src extrema(sum(abs2, gt_normal, dims=3)[mask])

if false # view angle of surface normal w.r.t. z-axis
    tmp = sqrt.(gt_normal[:,:,1].^2 + gt_normal[:,:,2].^2)
    tmp = rad2deg.(atan.(tmp, gt_normal[:,:,3]))
    jim(tmp)
end


function random_light(;
    rmax = 1/sqrt(2),
    θ = rand() * 2π,
)
#   r = rand()^0.5 * rmax
    r = 0.5
    x = r * cos(θ) + 0.1
    y = r * sin(θ) + 0.1
    z = sqrt(1 - x^2 - y^2)
    return [x, y, abs(z)]
end 

if !@isdefined(Ltrue)
end
    seed!(1)
    nlight = 20 # number of lighting directions
#   Ltrue = [random_light() for _ in 1:nlight]
    Ltrue = [random_light(; θ=il/nlight*1.5π) for il in 1:nlight]
    Ltrue = hcat(Ltrue...)'

tmp = range(0, 2π, 361)
plot(cos.(tmp), sin.(tmp), aspect_ratio=1,
    xaxis = (L"x", (-1,1)),
    yaxis = (L"y", (-1,1)),
    title = "Lighting directions",
)
pl_gt = scatter!(Ltrue[:,1], Ltrue[:,2], label = "True")

#prompt(); throw()

#=
Synthesize image data using Lambertian reflectance model
and examine singular values.
Ignoring self-shadows,
the rank is exactly 3.
With self-shadow effects
[Barsky & Petrou, 2003, T-PAMI](https://doi.org/10.1109/TPAMI.2003.1233898),
there are 3 dominant singular values
and additional small but non-negligible values.
=#

if !@isdefined(images)
end
    images_ideal = reshape(gt_normal, :, 3) * Ltrue' # hypothetical Lambertian
    images_ideal ./= maximum(images_ideal) # normalize
    svdval_ideal = svdvals(images_ideal)
    images = max.(images_ideal, 0) # "shadows" if lighting is ≥ 90° from normal
    svdval_images = svdvals(images)
    images = reshape(images, nx, ny, nlight)

pd = jim(images; title="Images for different lighting directions")

#src savefig(pd, "photometric3_data.pdf")

if !@isdefined(ps)
end
    ps = scatter(svdval_ideal, label="Ideal", color=:blue,
       xlabel=L"k", ylabel=L"σ_k", title="Singular values", size = (600,300))
    scatter!(ps, svdval_images, label="Realistic", color=:red)

    good = all(>(0), images, dims=3)
    images_good = reshape(images, :, nlight)[vec(good),:]
    svdval_good = svdvals(images_good)
    scatter!(ps, svdval_good, label="Good pixels", color=:green)
ps

#
prompt()

#src savefig(ps, "photometric3_svdvals.pdf")

#=
## Load "bunny" data used in
[2012 CVPR paper by Ikehata et al.](https://doi.org/10.1109/CVPR.2012.6247691).
This is a set of 50 images
under various lighting directions.
We use 25 of those images.

The "uncalibrated" approach used here
treats the lighting directions as being unknown,
unlike the
[original least-squares approach of Woodham, 1980](https://doi.org/10.1117/12.7972479).
=#

if !@isdefined(images) && false
    url0 = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/bunny_lambert/image000.npy"

    tmp = download(url0)
    x = npzread(tmp)

    dim = size(x)[1:2]
    nlight = 50
    images = zeros(Float32, dim..., nlight)
    images[:,:,1] = x[:,:,1]'

    for id in 2:nlight
        id3 = @sprintf("%03d", id-1)
        @show id3
        url1 = replace(url0, "000" => id3)
        xtmp = npzread(download(url1))
        images[:,:,id] = xtmp[:,:,1]'
    end

    i1 = 22:256-25
    i2 = 28:217
    il = [1:12; 30:40]
    images = images[i1,i2,il]
    images ./= maximum(images) # normalize
    dim = size(images)
## https://raw.githubusercontent.com/yasumat/RobustPhotometricStereo/master/data/bunny/mask.png
end
pd = jim(images; title="Images for different lighting directions")

#src savefig("photometric3_data.pdf")


#=
## Rank-3 approximation

For the Lambertian model,
we expect each pixel value
to be proportional
to the inner product
of the lighting direction
with the corresponding surface normal.

The surface normal is a 3-vector,
so we expect the images
to lie in a 3-dimensional subspace.
See
[Ikehata et al., CVPR 2012](https://doi.org/10.1109/CVPR.2012.6247691).

To make a low-rank approximation,
collect image data into a matrix
and then examine the singular values.

As expected,
the first 3 singular values are larger than rest.
(The matrix rank is not exactly 3 though.)
=#

#=
Estimate the lighting directions
using _only_ the pixels with no
=#

# The following permutation / sign-flip matrix was found empirically:
P = [0 0 -1; 0 1 0; 1 0 0]

light = svd(images_good).V[:,1:3]
light ./= sqrt.(sum(abs2, light, dims = 2)) # normalize to unit norm
light *= P # todo

tmp = deepcopy(pl_gt)
pl = scatter!(tmp, light[:,1], light[:,2], label = "Estimated")

plot(Ltrue)
scatter!(light)


#=

# The following permutation / sign-flip matrix was found empirically:
P = [0 0 -1; 0 1 0; 1 0 0]

# Examine estimated lighting directions
# from (permuted) first three right singular vectors.
L = V[:,1:3] # * P # notice permutation
L ./= sqrt.(sum(abs2, L, dims = 2)) # normalize to unit norm

pl = scatter(L[:,1], L[:,2], L[:,3],
   xaxis = (L"x", (-1, 1), (-1:1)*0.8),
   yaxis = (L"y", (-1, 1), (-1:1)*0.8),
   zaxis = (L"z", (0.2, 1.0), 0.2:0.1:1.0),
   title = "Estimated lighting directions",
)

#
prompt()


#=
Next we examine the estimated surface normals.
Those estimates are meaningful
only where the object is present,
so first we determine an object "mask".
=#
mask = sum(images, dims = 3) .> eps(Float32)
pm = jim(mask, "Mask")


if !@isdefined(gt_lights)
end
    url = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/lights.npy"
    gt_lights = npzread(download(url))
    gt_lights = gt_lights[il,:]
#src extrema(sum(abs2, gt_lights, dims=2))
pl_gt = scatter(eachslice(gt_lights, dims=2)...,
   xaxis = (L"x", (-0.8, 0.8), (-1:1)*0.8),
   yaxis = (L"y", (-0.8, 0.8), (-1:1)*0.8),
   zaxis = (L"z", (0.6, 1.0), 0.6:0.1:1.0),
   title = "True lighting directions",
)

plot(pl, pl_gt)

# Estimated surface normals
normals = U[:,1:3] * P' # P is a unitary matrix
normals ./= sqrt.(sum(abs2, normals, dims=2)) # normalize
normals = reshape(normals, dim[1:2]..., 3)
normals .*= mask # apply mask
pn = jim(normals; nrow=1, title="Estimated surface normals")

#src savefig("photometric3_pn1.pdf")


#=
It is conventional in the photometric stereo literature
to show the surface normals using a color image.
=#

using ImageCore: colorview
using Colors: RGB
colorview(RGB, eachslice(normals, dims=3)...)

#
prompt()
throw()


include("../../../inc/reproduce.jl")
=#
