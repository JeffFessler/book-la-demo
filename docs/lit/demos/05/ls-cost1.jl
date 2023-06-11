#=
# [LS cost functions](@id ls-cost1)

This example illustrates linear least-squares (LS) cost functions
and minimum-norm LS (MNLS) solutions
using the Julia language.
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
        "LaTeXStrings"
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
        "Random"
    ])
end


# Tell this Julia session to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using InteractiveUtils: versioninfo
using LaTeXStrings
using LinearAlgebra: norm, pinv
using MIRTjim: prompt
using Plots: default, contour, plot!, scatter!, savefig
using Random: seed!
default(); default(label="", markerstrokecolor=:auto, markersize=6, linewidth=2,
    xlims = (-3,3), ylims = (-3,3), aspect_ratio=:equal, size=(450,400),
    legendfontsize=12, guidefontsize=13, tickfontsize=10, labelfontsize=18)

# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Under-determined case

A = [1 2] # 1st case: M < N
y = [3] # obviously x=[1 1] is one possible solution (but not MNLS)
f1(x) = norm(A*x - y)
xh1 = A \ y

x1 = range(-1,1,101) * 3
x2 = range(-1,1,103) * 3
c1 = [f1([x1a, x2a]) for x1a in x1, x2a in x2];

flabel(n) = L"\{%$bx : y_{%$n} = %$(bA)_{[%$n,:]} %$bx\}"
bx = "\\mathit{\\mathbf{x}}"
by = "\\mathit{\\mathbf{y}}"
bA = "\\mathit{\\mathbf{A}}"
color = :viridis
p1 = contour(x1, x2, c1', label="contours"; color)
plot!(xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
scatter!([0], [0], color=:black, markershape=:square)
plot!([0, xh1[1]], [0, xh1[2]], line=:magenta)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash),
    label=L"\{%$bx : %$by = %$bA %$bx\}")
scatter!([1], [1], color=:blue, markershape=:star5, label=L"(1,1)")
scatter!([xh1[1]], [xh1[2]], color=:red, markershape=:circle, label="MNLS",
    title = "Under-determined case")

#
prompt()

## savefig(p1, "demo_ls_cost1a.pdf")


# ## Square but singular case

A = [1 2; 2 4] # 2nd case: M = N but singular A
y = [3, 6] # again [1,1] is a solution (but not MNLS solution)
f2(x) = norm(A*x - y)
xh2 = pinv(A) * y

c2 = [f2([x1a, x2a]) for x1a in x1, x2a in x2]
p2 = contour(x1, x2, c2', label="contours"; color)
plot!(xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
scatter!([0], [0], color=:black, markershape=:square)
plot!([0, xh2[1]], [0, xh2[2]], line=:magenta)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label=flabel(1))
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dashdot), label=flabel(2))
scatter!([1], [1], color=:blue, markershape=:star5, label=L"(1,1)")
scatter!([xh2[1]], [xh2[2]], color=:red, markershape=:circle, label="MNLS",
 title = "Singular case")

#
prompt()

## savefig(p2, "demo_ls_cost1b.pdf")


# ## Square non-singular case

A = [1 2; 1 3] # 3rd case: M = N with non-singular A
y = [3, 4] # now x=[1,1] is the unique solution (by design)
f3(x) = norm(A*x - y)
xh3 = A \ y

c3 = [f3([x1a, x2a]) for x1a in x1, x2a in x2]
p3 = contour(x1, x2, c3', label="contours"; color)
plot!(xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label=flabel(1))
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dash), label=flabel(2))
scatter!([xh3[1]], [xh3[2]], color=:red, markershape=:circle, label="LLS",
 title = "Non-singular case")

#
prompt()

## savefig(p3, "demo_ls_cost1c.pdf")


# ## Typical over-determined case

A = [1 2; 1 -1; 2 1] # 4th case: M > N with (typical) inconsistent data
y = [3, 2, 1] # no consistent solution
f4(x) = norm(A*x - y)
xh4 = A \ y

c4 = [f4([x1a, x2a]) for x1a in x1, x2a in x2]
p4 = contour(x1, x2, c4', label="contours"; color)
plot!(xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label=flabel(1))
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dash), label=flabel(2))
plot!(x1, (y[3] .- A[3,1]*x1)/A[3,2], line=(:purple,:dash), label=flabel(3))
scatter!([xh4[1]], [xh4[2]], color=:red, markershape=:circle, label="LLS",
    title = "Over-determined case")

#
prompt()

## savefig(p4, "demo_ls_cost1d.pdf")


include("../../../inc/reproduce.jl")
