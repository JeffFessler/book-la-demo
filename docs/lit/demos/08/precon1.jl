#=
# [Preconditioning](@id precon1)

This example illustrates
the effects of preconditioning matrices
for gradient descent (GD) for least squares (LS) problems,
using the Julia language.
* 2019-11-19 Created by Steven Whitaker  
* 2023-05-30 Julia 1.9 by Jeff Fessler
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

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: svd, norm, svdvals, eigvals, Diagonal, I
using MIRTjim: prompt
using Plots: contour, default, gui, plot, plot!, savefig, scatter!
using Random: seed!
default(); default(markerstrokecolor=:auto, label = "", markersize=6,
 tickfontsize=12, labelfontsize=18, legendfontsize=18)


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);



#=
## Background

The cost function to minimize for least squares problems is
``f(x) = \frac{1}{2} ‖ A x - y ‖_2^2,``
and its gradient is
``∇ f(x) = A' (A x - y).``

Preconditioned GD with positive definite preconditioner
``P`` has the following update:

``x_{k+1} = x_{k} - P A' (A x_k - y).``

For preconditioned GD to converge from any starting point,
the following must be satisfied:

``-1 < \mathrm{eig}\{G\} < 1``,
where
``G = I - P^{1/2} A' A P^{1/2}.``

Furthermore, the closer the eigenvalues of ``G`` are to zero,
the faster preconditioned GD converges.

## Setup

This notebook creates a matrix
``A \in \mathbb{R}^{3 \times 2}``
with specified singular values,
and uses
``y = 0 \in \mathbb{R}^3``
for simplicity of plots.

Create random `3 × 2` matrix
with singular values 10 and 3
=#

seed!(0)
(U, _, V) = svd(randn(3,2))
A = U * [10 0; 0 3] * V';

# Set up LS cost function and its gradient
f(x) = 0.5 * norm(A * x)^2
∇f(x) = A' * (A * x)

# Function for generating the matrix ``G`` from a given preconditioner matrix
G(sqrtP) = I - sqrtP' * (A' * A) * sqrtP;


#=
## Gradient Descent

First consider regular GD,
i.e., preconditioned GD with
``P = \alpha I,``
where ``α`` is the step size.  

We use the optimal step size
``α = \frac{2}{σ_1^2(A) + σ_N^2(A)}.``

Pick step size (preconditioner ``P = αI``)
=#
α = 2 / (sum(svdvals(A' * A)[[1, end]])) # Optimal step size
eigvals(G(sqrt(α))) # Eigenvalues of G govern rate of convergence

# Plot cost function
x1 = -10:0.1:10
x2 = -10:0.1:10
xidx = [[x1[i], x2[j]] for i in 1:length(x1), j in 1:length(x2)]
scale = 1/1000 # simplify clim
pu = contour(x1, x2, scale * f.(xidx)', annotate = (1, 6, "Unpreconditioned"),
 xaxis = (L"x_1", (-1,1).*10),
 yaxis = (L"x_2", (-1,1).*10),
 size = (500,400),
);

x0 = [5.0, -8.0] # initial guess
scatter!(pu, [x0[1]], [x0[2]], color=:green, label = L"x_0");

# Run GD
niter = 100
x = Vector{Vector{Float64}}(undef, niter + 1)
x[1] = x0
for k in 1:niter
    x[k+1] = x[k] - α * ∇f(x[k])
end

# Display iterates
plot!(pu, [x[k][1] for k in 1:niter], [x[k][2] for k in 1:niter],
    marker=:star, color=:blue, label = L"x_k");

# Mark the minimum of the cost function
scatter!(pu, [0], [0], label = L"\hat{x}", color=:red,
    aspect_ratio = :equal, marker = :x)

#src savefig(pu, "precon1-pu.pdf")

#
prompt()


#=
The contours of our cost function ``f(x)`` are ellipses.
The ratio of the singular values of ``A`` determines
the eccentricity, or how oblong (non-circular) the ellipse is.
In our case, the singular values are 10 and 3,
so the major axis of the contour ellipse is 10/3 as long as the minor axis.

## Preconditioned Gradient Descent

Now let's see how adding a preconditioner matrix changes things.

Manipulating the preconditioned GD step for LS problems
leads to the following update:

``x_{k+1} = x_k - P A' (A x_k - y)``

``P^{-1/2} x_{k+1} = P^{-1/2} x_k - P^{1/2} A' (A x_k - y)``

``P^{-1/2} x_{k+1} = P^{-1/2} x_k - P^{1/2} A' (A P^{1/2} P^{-1/2} x_k - y)``

``z_{k+1} = z_k - P^{1/2} A' (A P^{1/2} z_k - y)``, where ``z_k = P^{-1/2} x_k``

``z_{k+1} = z_k - \tilde{A}' (\tilde{A} z_k - y)``,
where ``\tilde{A} = A P^{1/2}``,
and we used the fact that ``P^{1/2}`` is Hermitian symmetric.

This last equation is the normal (not preconditioned) GD step
(with step size 1)
for a LS problem with cost function
``\tilde{f}(z) = \frac{1}{2} ‖ \tilde{A} z - y ‖_2^2``.

## Clicker Question
The preconditioned LS cost function ``\tilde{f}``
relates to the non-preconditioned LS cost function ``f``
via the relation ``\tilde{f}(z) = f(g(z))``
for what function ``g``?

* A. ``g(z) = P^{-1/2} z``
* B. ``g(z) = P^{1/2} z``
* C. ``g(z) = z``
* D. ``g(z) = \tilde{A} z``
* E. ``g(z) = \tilde{A}' z``


## Ideal Preconditioner

We first consider the ideal preconditioner
``P = (A' A)^{-1}``.

Compute ideal preconditioner
=#
sqrtPideal = sqrt(inv(A' * A))
eigvals(G(sqrtPideal))

# Set up preconditioned cost function and its gradient
f̃ideal(z) = f(sqrtPideal * z)
∇f̃ideal(z) = sqrtPideal' * ∇f(sqrtPideal * z);

# Plot preconditioned cost function
z1 = -40:40
z2 = -40:40
zidx = [[z1[i], z2[j]] for i in 1:length(z1), j in 1:length(z2)]
scale = 1/250 # simplify clim
ph = contour(z1, z2, scale * f̃ideal.(zidx)',
 annotate = (9, 24, "Ideal preconditioner"),
 xaxis = (L"z_1", (-1,1).*40),
 yaxis = (L"z_2", (-1,1).*40),
 size = (500,400),
);

# Transform initial x guess into z coordinates and plot
z0 = sqrtPideal \ x0
scatter!(ph, [z0[1]], [z0[2]], color=:green, label = L"z_0");

# Run GD
zk = z0 - ∇f̃ideal(z0)

# Display iterates
plot!(ph, [z0[1],zk[1]], [z0[2],zk[2]], marker=:star, color=:blue, label = L"z_k");

# Mark the minimum of the preconditioned cost function
scatter!(ph, [0], [0], label = L"\hat{z}", color=:red,
    aspect_ratio = :equal, marker = :x)

#src savefig(ph, "precon1-ph.pdf")

#
prompt()

#=
Using the ideal preconditioner caused a coordinate change
in which the contours of our cost function are circles.
In this new coordinate system,
the negative gradient of our cost function points towards the minimizer.
Furthermore, with the ideal preconditioner GD converged in just one step,
which agrees with the fact that the eigenvalues of ``G``
for this preconditioner are 0
(ignoring numerical precision issues).
Unfortunately, computing the ideal preconditioner is expensive.

## Diagonal Preconditioner

A less expensive preconditioner is the diagonal preconditioner
``P = \alpha \; \mathrm{diag}\{|A' A| 1_N\}^{-1}``.  
For convergence, we must have ``0 < \alpha < 2``.
We use an empirically chosen value for ``α``
in that range.

Pick step size and compute diagonal preconditioner
=#
α = 1.71 # Chosen empirically
sqrtPdiag = sqrt(α * inv(Diagonal(abs.(A' * A) * ones(size(A, 2)))))
eigvals(G(sqrtPdiag))

# Set up preconditioned cost function and its gradient
f̃diag(z) = f(sqrtPdiag * z)
∇f̃diag(z) = sqrtPdiag' * ∇f(sqrtPdiag * z);

# Plot preconditioned cost function
z1 = -50:50
z2 = -50:50
zidx = [[z1[i], z2[j]] for i in 1:length(z1), j in 1:length(z2)]
scale = 1/500 # simplify clim
pd = contour(z1, z2, scale * f̃diag.(zidx)',
 annotate = (12, 30, "Diagonal preconditioner"),
 xaxis = (L"z_1", (-1,1).*50),
 yaxis = (L"z_2", (-1,1).*50),
 size = (500,400),
);

# Transform initial x guess into z coordinates and plot
z0 = sqrtPdiag \ x0
scatter!(pd, [z0[1]], [z0[2]], color=:green, label = L"z_0");

# Run GD
niter = 100
z = Array{Array{Float64,1},1}(undef, niter + 1)
z[1] = z0
for k in 1:niter
    z[k+1] = z[k] - ∇f̃diag(z[k])
end;

# Display iterates
plot!(pd, [z[k][1] for k in 1:niter], [z[k][2] for k in 1:niter],
    marker=:star, color=:blue, label = L"z_k");

# Mark the minimum of the preconditioned cost function
scatter!(pd, [0], [0], label = L"\hat{z}", color=:red,
    aspect_ratio = :equal, marker = :x)

#src savefig(pd, "precon1-pd.pdf")

#
prompt()

#=
Using the diagonal preconditioner did cause a coordinate change,
but one less dramatic than did the ideal preconditioner.
The contours in this new coordinate system are still ellipses,
but they are slightly more circular.
Using the diagonal preconditioner also resulted in eigenvalues of ``G``
that are smaller than when
using (non-preconditioned) GD with optimal step size,
and one can see that using the diagonal preconditioner appears
to converge more quickly.

The following reports the ratio of the singular values
of the three different ``A`` (or ``\tilde{A}``) matrices used here.
A value of 1 corresponds to circular cost function contours,
and higher values correspond to more elliptical contours.
=#

"Ratio of singular values of A, A * sqrtPideal A * sqrtPdiag:"

[
/(svdvals(A)...)
/(svdvals(A * sqrtPideal)...)
/(svdvals(A * sqrtPdiag)...)
]

# Here are the three plots displayed next to each other.

pp = plot(
    plot!(pu, title = "GD"),
    plot!(ph, title = "Ideal"),
    plot!(pd, title = "Diagonal"),
    size = (1900,470),
    layout=(1,3),
)

#src savefig(pp, "precon1-pp.pdf")

include("../../../inc/reproduce.jl")
