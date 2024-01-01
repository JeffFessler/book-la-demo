#=
# [Kronecker sum of circulant](@id kron-sum-inv)

This example illustrates
efficient computation
of the inverse of a Kronecker sum
of circulant matrices
using the Julia language.

Related to Problem 8.7 in the textbook.
=#

#srcURL

#src based on hp123.jl

#=
## Setup
Add the Julia packages used in this demo.
Change `false` to `true` in the following code block
if you are using any of the following packages for the first time.
=#

if false
    import Pkg
    Pkg.add([
        "FFTW"
        "InteractiveUtils"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using FFTW: fft, ifft
using InteractiveUtils: versioninfo
using LinearAlgebra: I
using MIRTjim: jim, prompt
using Plots: savefig
using Random: seed!
seed!(0)


# The following line helps when running this jl-file as a script;
# this way it prompts user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# Circulant matrix with given first column
function circm(x)
    N = length(x)
    return hcat([circshift(x, k-1) for k in 1:N]...)
end

#=
## Numerical test
First perform a numerical test to verify the formulas.
=#

# Test data
M, N = 64, 32
b = randn(N)
c = randn(M)
X = randn(M, N);

# Matrix-inverse solution
B = circm(b) # N
C = circm(c) # M
A = kron(B, I(M)) + kron(I(N), C)
y = inv(A) * vec(X);

# Denominator
P = fft(c) .+ transpose(fft(b));

# DFT solution
Fn = fft(I(N), 1)
Fm = fft(I(M), 1)
Y1 = (Fm' * ((Fm * X * transpose(Fn)) ./ P) * conj(Fn)) / (M * N)
@assert Y1 ≈ real(Y1) # should be real-valued
Y1 = real(Y1); # so discard the imaginary part

# FFT solution
Y2 = ifft(fft(X) ./ P)
@assert Y2 ≈ real(Y2)
Y2 = real(Y2);

# Verify
@assert y ≈ vec(Y1)
@assert y ≈ vec(Y2)


#=
## Image data
Now illustrate visually
using additively separable blur
with individually invertible blur kernels.
=#

X = ones(60, 64); X[20:50,10:50] .= 2
M, N = size(X)
b = zeros(N); b[1 .+ mod.(-3:3,N)] = (4 .- abs.(-3:3)); b[1] +=  1
b /= sum(b)
c = zeros(M); c[1 .+ mod.(-4:4,M)] = (5 .- abs.(-4:4)); c[1] +=  1
c /= sum(c)
B = circm(b)
C = circm(c)
pb = jim(B, "circulant B", size=(300,300))
pc = jim(C, "circulant C", size=(300,300))
p1 = jim(pb, pc; size=(600,300))

#
prompt()


p2 = jim(X; title="X original", size=(300,300))
Y = C * X + X * transpose(B)
p3 = jim(Y; title="Y blurred", size=(300,300))
p23 = jim(p2, p3; size=(600,300))

#
prompt()


# FFT solution
P = fft(c) .+ transpose(fft(b)) # denominator
Xhat = ifft(fft(Y) ./ P)
@assert Xhat ≈ real(Xhat)
Xhat = real(Xhat)
@assert Xhat ≈ X

include("../../../inc/reproduce.jl")
