#=
# [Convolution matrix](@id conv-mat)

This example illustrates
1D signal convolution
represented as matrix operations
(for various boundary conditions)
using the Julia language.
=#

#src based on book/c-restore/fig/fig_res_mat1.m
#src based on book/c02mat/fig/conv-mat.jl

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
        "ColorSchemes"
        "DSP"
        "FFTW"
        "FFTViews"
        "InteractiveUtils"
        "LaTeXStrings"
        "LinearMapsAA"
        "MIRTjim"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using ColorSchemes
using DSP: conv
using FFTW: fft, ifft
using FFTViews: FFTView
using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearMapsAA
using MIRTjim: jim
using Plots: RGB, default, gui, heatmap, plot, plot!, savefig, scatter!, @layout
using Plots.PlotMeasures: px
default(markerstrokecolor=:auto, markersize=11, linewidth=5, label="",
 tickfontsize = 11, labelfontsize = 15, titlefontsize=18)


#=
## Filter
Define a 5-tap
finite impulse response (FIR) filter
``h[k]`` with support `-2:2`
and display it.
=#

psf = [1, 3, 5, 4, 2] # filter impulse response
K = length(psf)
hp = Int((K-1)/2) # 2 (half width)

cols = [ColorSchemes.viridis[x] for x in range(0,1,K)]

pp = plot(widen=true, tickfontsize = 14, labelfontsize = 20,
    xaxis = (L"k", (-1,1) .* 4, -4:4),
    yaxis = (L"h[k]", (-0.2, 5.2)),
    size = (600, 250), left_margin = 15px, bottom_margin = 18px,
)
plot!([-5, 5], [0, 0], color=:black, linewidth=2)
for k in 1:K
    c = psf[k]
    c = cols[c]
    plot!([k-hp-1], [psf[k]], line=:stem, marker=:circle, color=c)
end
for k in (hp+1):4
    scatter!([k], [0], color=:grey)
    scatter!([-k], [0], color=:grey)
end
gui()
## savefig(pp, "psf.pdf")


#=
## Convolution matrices

Define convolution matrices
for various end conditions.
The names of the conditions
match those of Matlab's `conv` function.
=#

N = 10 # input signal length
kernel = FFTView(zeros(Int, N))
kernel[-hp:hp] = psf;


#=
### `'full'` for zero end conditions
where the output signal length is
``M = N + K - 1``.
=#
Az = LinearMapAA(x -> conv(x, psf), (N+K-1, N))
Az = Matrix(Az)
Az = round.(Int, Az) # because `conv` apparently uses `fft`
size(Az)


#=
### 'circ' for periodic end conditions
for which ``M = N``.
=#
Ac = LinearMapAA(x -> ifft(fft(x) .* fft(kernel)), (N,N) ; T = ComplexF64)
Ac = Matrix(Ac)
Ac = round.(Int, real(Ac))
size(Ac)


#=
### 'same' for matching input and output signal lengths,
so ``M = N``.
=#
As = Az[hp .+ (1:N),:]
size(As)


#=
### 'valid'
where output is defined only for samples
where the shifted impulse overlaps the input signal,
for which ``M = N-K+1``.
=#
Av = Az[(K-1) .+ (1:(N-(K-1))),:]
size(Av)


#=
### Display convolution matrices
using colors that match the 1D impulse response plot.
=#
cmap = [RGB{Float64}(1.,1.,1.) .* 0.8; cols];

#src jy = (x,t,y) -> jim(x', t, color=cmap, ylabel=y)
jy = (x,t,y) -> jim(x'; color=cmap, ylims=(0.5,14.5), ylabel=y, labelfontsize=12);

pz = jy(Az, "'full'", L"M=N+K-1")
pv = jy(Av, "'valid'", L"M=N-K+1")
ps = jy(As, "'same'", L"M=N")
pc = jy(Ac, "circulant", L"M=N")
## plot(pz, pv, ps, pc; size=(400,400))
p4 = plot(pz, pv, ps, pc; size=(1000,200), layout=(1,4), left_margin = 20px)

## savefig(p4, "match4.pdf")
## savefig(pz, "a-full.pdf")
## savefig(pv, "a-valid.pdf")
## savefig(ps, "a-same.pdf")
## savefig(pc, "a-circ.pdf")

l = @layout [
 a{0.3w} b{0.3w}
 c{0.3w} d{0.3w}
]
p22 = plot(pz, pv, ps, pc) #, layout = l, size = (500,500))
## savefig(p22, "all4.pdf")


include("../../../inc/reproduce.jl")
