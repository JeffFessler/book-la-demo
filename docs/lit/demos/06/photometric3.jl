#=
# [Photometric stereo](@id photometric3)

This example illustrates
[photometric stereo](https://en.wikipedia.org/wiki/Photometric_stereo)
for
[Lambertian surfaces](https://en.wikipedia.org/wiki/Lambertian_reflectance)
using the Julia language.

This method determines the surface normals of an object
from 3 or more pictures of the object
taken with different lighting directions.

This demo follows the "uncalibrated" approach of
[Hayakawa, JOSA, 1994](https://doi.org/10.1364/JOSAA.11.003079)
that treats the lighting directions as being unknown,
unlike
the original least-squares approach of
[Woodham, 1980](https://doi.org/10.1117/12.7972479).
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
        "Downloads"
        "InteractiveUtils"
        "LinearAlgebra"
        "LaTeXStrings"
        "MIRTjim"
        "NPZ"
        "Plots"
        "Printf"
        "Random"
    ])
end


# Tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using Downloads: download
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: Diagonal, svd, svdvals, rank, norm, pinv
using MIRTjim: jim, prompt
using NPZ: npzread
using Plots: plot, scatter, scatter!, ylims!, cgrad, default, RGB, savefig
    default(label="", markerstrokecolor=:auto),
using Printf: @sprintf
using Random: seed!


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


#=
## Ground-truth surface normals

Load ground truth surface normal vectors of "bunny" data used in
[2012 CVPR paper by Ikehata et al.](https://doi.org/10.1109/CVPR.2012.6247691).
=#
if !@isdefined(gt_normal_bunny)
    url = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/gt_normal.npy"
    tmp = npzread(download(url))
    i1, i2 = 32:256-25, 33:212 # crop to reduce compute
    tmp = permutedims(tmp, [2, 1, 3]) # "transpose"
    gt_normal_bunny = tmp[i1, i2, :] # crop
end;

#=
Create hemisphere to augment the bunny data.
=#
function hemi_normal(x, y ; # surface normal of a hemi-ellipsoid
    xh = 20f0, yh = xh, zh = xh,
)
    tmp = (x/xh)^2 + (y/yh)^2
    if tmp < 1
        z = zh * sqrt(1 - tmp)
        tmp = [x/xh^2, y/yh^2, z/zh^2]
        return tmp / norm(tmp)
    end
    return [0, 0, 0]
end;

if true
    rh = 20
    x = -rh:rh
    y = x
    tmp = hemi_normal.(x, y'; xh=rh)
    tmp = reduce(hcat, tmp)
    tmp = reshape(tmp', 2rh+1, 2rh+1, 3)
end;

#=
Define ground truth normals
as combination of bunny normals
and hemisphere normals.
=#
if !@isdefined(gt_normal)
    gt_normal = copy(gt_normal_bunny)
    gt_normal[176 .+ x, 25 .+ y, :] = tmp
    nx, ny = size(gt_normal)[1:2]
    shape2 = x -> reshape(x, prod(size(x)[1:2]), :)
    shape3 = x -> reshape(x, nx, ny, :)
end;

# The three images are the x, y, and z components
pn_gt = jim(gt_normal; title="Ground-truth normals", nrow=1,
    xaxis=L"x", yaxis=L"y", size=(600,300), clim=(-1,1), colorbar_ticks=-1:1)


#=
Surface normals are meaningful
only where the object is present,
so we determine an object "mask".
=#
mask = dropdims(sum(abs, gt_normal, dims=3), dims=3) .> eps(Float32)
pm = jim(mask, "Mask"; cticks=0:1)

# Verify that the surface normals are unit norm
# (within the mask).
@assert maximum(abs, sum(abs2, gt_normal, dims=3)[vec(mask)] .- 1) < 1e-12

#=
View angle of surface normal w.r.t. z-axis
=#
if true
    tmp = sqrt.(sum(abs2, gt_normal[:,:,1:2]; dims=3))
    tmp = rad2deg.(atan.(tmp, gt_normal[:,:,3]))
    pza = jim(tmp; title="Angle of surface normal w.r.t. z axis",
        ctitle="degrees", cticks=0:30:90)
end


#=
## Lighting directions

Define lighting directions
for simulated object views.
=#
function light_vector( ;
    θ = rand() * 2π,
    r = 1/sqrt(2),
    xoffset = 0.2,
    yoffset = 0.2,
    x = r * cos(θ) + xoffset,
    y = r * sin(θ) + yoffset,
)
    z = sqrt(1 - x^2 - y^2)
    return [x, y, z]
end;

# Deliberately asymmetric directions to aid testing
if !@isdefined(Ltrue)
    nlight = 12 # number of lighting directions
    Ltrue = [light_vector(; θ=il/nlight*1.7π, r=0.5-0.1*il/nlight) for il in 1:nlight]
    Ltrue = reduce(hcat, Ltrue) # (3, nlight)
    @assert maximum(abs, sum(abs2, Ltrue; dims=1) .- 1) < 9eps()
    @show extrema(Ltrue[3,:])
end;

# Plot lighting directions
tmp = range(0, 2π, 361)
plot(cos.(tmp), sin.(tmp); aspect_ratio=1, color=:black,
    xaxis = (L"x", (-1,1)),
    yaxis = (L"y", (-1,1)),
    title = "Lighting directions",
)
pl_gt = scatter!(Ltrue[1,:], Ltrue[2,:]; label = "True", color=:red)

#
prompt()


#=
## Synthesize images

Synthesize image data using Lambertian reflectance model.

For the Lambertian model,
each pixel value
is proportional
to the inner product
of the lighting direction
with the corresponding surface normal.

If that inner product is zero for some pixels,
then the image contains "shadows" at those pixels.
[Wu et al., ACCV, 2011](https://doi.org/10.1007/978-3-642-19318-7_55)
describe the `max(⋅,0)` operation below as "attached shadows."
=#

if !@isdefined(images)
    images_ideal = shape2(gt_normal) * Ltrue # hypothetical Lambertian
    images_ideal ./= maximum(images_ideal) # normalize
    svdval_ideal = svdvals(images_ideal)
    images = max.(images_ideal, 0) # "shadows" if lighting is ≥ 90° from normal
    svdval_images = svdvals(images)
    images = shape3(images)
end;

# Note the different shadings in the different images.
# Obviously the bunny cannot "jump around" during the imaging...
pd = jim(images; title="Images for $nlight different lighting directions",
    caxis=("Intensity", (0,1), 0:1))

#src savefig(pd, "photometric3_data.pdf")


#=
## Low-rank structure

Examine the singular values.
Ideally
(i.e., ignoring shadows),
there would be (at most) 3 nonzero singular values,
because `images_ideal`
is the product of (`npixel` × 3) object normal matrix
with a (3 × `nlight`) lighting direction matrix,
so its rank is at most 3.

With self-shadow effects, e.g.,
[Barsky & Petrou, 2003, T-PAMI](https://doi.org/10.1109/TPAMI.2003.1233898),
there are 3 dominant singular values
and additional small but non-negligible values.
The `images` matrix rank is not exactly 3
because of the shadow effects.
=#

if !@isdefined(ps)
    ps1 = scatter(svdval_ideal, label="Ideal", color=:blue,
       xaxis=(L"k", (1,nlight), [1,3,4,nlight]),
       yaxis=(L"σ_k",), marker=:x, widen=true,
       title="Singular values")
    scatter!(ps1, svdval_images, label="Realistic", color=:red)

    good = all(>(0), images, dims=3)
    images_good = shape2(images)[vec(good),:]
    svdval_good = svdvals(images_good)
    scatter!(ps1, svdval_good, label="Good pixels", color=:green)
    ps4 = deepcopy(ps1)
    ylims!(ps4, (0,5); title="zoom")
    ps = plot(ps1, ps4)
end

#
prompt()

#src savefig(ps, "photometric3_svdvals.pdf")



#=
## Rank-3 approximation

To make a low-rank approximation,
collect image data into a `npixel × nlight` matrix
and use a SVD.

Estimate the lighting directions
using _only_ the pixels with no shadows.
=#

tmp = svd(images_good)
light1 = tmp.Vt[1:3,:] # right singular vectors
normal1 = tmp.U[:,1:3] * Diagonal(tmp.S[1:3])
@assert norm(images_good - normal1 * light1) / norm(images_good) < 9eps()


#=
## Estimate lighting directions

Apply method of
[Hayakawa, JOSA, 1994](https://doi.org/10.1364/JOSAA.11.003079)
to resolve non-uniqueness issue,
under the simplifying assumption
(satisfied here)
that the light intensity is the same
for all lighting directions.

That method does not explicitly exploit
the fact that ``A'A`` is positive semi-definite.
Challenge:
develop method that does use that property.
=#

Abig = reduce(vcat, map(c -> kron(c', c'), eachcol(light1)))
tmp = Abig \ ones(nlight)
B = reshape(tmp, 3, 3) # B = A'A
@assert B ≈ B' # (symmetry check)
#src tmp = svd(B)
#src @assert tmp.U ≈ tmp.V # (symmetry)
A = sqrt(B)
@assert A'A ≈ B
light2 = A * light1
@assert maximum(abs, sum(abs2, light2, dims=1) .- 1) < 30eps()

normal2 = normal1 * inv(A)
@assert norm(images_good - normal2 * light2) / norm(images_good) < 10eps()
@assert maximum(abs, norm.(eachrow(normal2)) .- 1) < 9e-6 # already unit norm!

#=
As described in
[Hayakawa, JOSA, 1994](https://doi.org/10.1364/JOSAA.11.003079),
the estimated lighting and surface normals
are in an arbitrary 3D coordinate system.
To display them in a useful way,
we use the Procrustes method
to align the coordinate system
with that of the original lighting.
=#
if true
    tmp = Ltrue[:,1:3] * light2[:,1:3]' # use just the first 3 sources
    tmp = svd(tmp)
    tmp = tmp.U * tmp.Vt
    light3 = tmp * light2
    normal3 = normal2 * tmp'
    @assert norm(images_good - normal3 * light3) / norm(images_good) < 10eps()
end

#=
Plot estimated lighting directions
(after coordinate system alignment)
=#
tmp = deepcopy(pl_gt)
pl = scatter!(tmp, light3[1,:], light3[2,:],
    marker = :x, label = "Estimated", color=:blue)

#
prompt()


#=
## Estimate surface normals.

Now that we have estimated the lighting directions,
return to estimate the surface normals
for _all_ pixels,
not just the "good" pixels.
=#
normal3 = shape3(shape2(images) * pinv(light3));

#=
Examine the estimated surface normals.
The accuracy is very good,
except in the shadow regions.
=#
pn_hat = jim(normal3; nrow=1, title="Estimated normals",
    clim=(-1,1), colorbar_ticks=-1:1)
RGB255(args...) = RGB((args ./ 255)...)
color = cgrad([RGB255(230, 80, 65), :black, RGB255(23, 120, 232)])
pn = jim(
 pn_gt,
 pn_hat,
 jim(normal3 - gt_normal; nrow=1, title="Difference", color,
     clim=(-1,1), colorbar_ticks=-1:1);
 layout=(3,1),
)

#src savefig(pn, "photometric3_pn1.pdf")


#=
This demo illustrates the utility of the SVD
and low-rank matrix approximation.

More advanced methods
handle shadows by allowing sparse errors, e.g.,
* [Wu et al., ACCV 2011](https://doi.org/10.1007/978-3-642-19318-7_55)
* [Ikehata et al., CVPR 2012](https://doi.org/10.1109/CVPR.2012.6247691),
or handle more general lighting conditions, e.g.,
* [Basri & Jacobs, CVPR 2001](https://doi.org/10.1109/CVPR.2001.990985).
=#


#=
## Exercise

Apply the method described above
to the bunny data used in
[Ikehata et al., CVPR 2012](https://doi.org/10.1109/CVPR.2012.6247691).
This is a set of 50 images
under various lighting directions.

As a starting point,
here we load that data.
=#

if !@isdefined(images_bunny)
    url0 = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/bunny_lambert/image000.npy"

    index_bunny = 0:5:45 # just load 10 of the 50
    nlight_bunny = length(index_bunny)
    tmp = download(url0)
    x = npzread(tmp)

    dim = size(x)[1:2]
    images_bunny = zeros(Float32, dim..., nlight_bunny)
    images_bunny[:,:,1] = x[:,:,1]'

    for (iz, index) in enumerate(index_bunny[2:end])
        id3 = @sprintf("%03d", index)
        @show id3
        url1 = replace(url0, "000" => id3)
        xtmp = npzread(download(url1))
        images_bunny[:,:,iz+1] = xtmp[:,:,1]'
    end
    images_bunny ./= maximum(images_bunny) # normalize
end
pb = jim(images_bunny; title="Images for different lighting directions")


#=
For reference,
here are the ground truth lighting directions.
=#
if !@isdefined(gt_lights)
    url = "https://github.com/yasumat/RobustPhotometricStereo/raw/master/data/bunny/lights.npy"
    gt_lights = npzread(download(url))
    gt_lights = gt_lights'[:,index_bunny .+ 1]
end
pl_gtb = scatter(eachrow(gt_lights)...,
   xaxis = (L"x", (-0.8, 0.8), (-1:1)*0.8),
   yaxis = (L"y", (-0.8, 0.8), (-1:1)*0.8),
   zaxis = (L"z", (0.4, 1.0), [0.4, 0.69, 0.96]),
   title = "True lighting directions",
)

#
prompt()

#=
If needed, here is the url for the mask:
* https://raw.githubusercontent.com/yasumat/RobustPhotometricStereo/master/data/bunny/mask.png
=#


#src #=
#src It is conventional in the photometric stereo literature
#src to show the surface normals using a color image.
#src =#

#src using ImageCore: colorview
#src using Colors: RGB
#src colorview(RGB, eachslice(normals, dims=3)...)

include("../../../inc/reproduce.jl")
