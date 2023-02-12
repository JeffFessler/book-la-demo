#=
# [LS cost functions](@id ls-cost1)

This example illustrates linear least-squares (LS) cost functions
and minimum-norm LS (MNLS) solutions
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
        "LinearAlgebra"
        "Plots"
        "LaTeXStrings"
        "MIRTjim"
        "Random"
        "InteractiveUtils"
    ])
end


# Now tell this Julia session to use the following packages for this example.
# Run `Pkg.add()` in the preceding code block first, if needed.

using LinearAlgebra: norm, pinv
using Random: seed!
using Plots; default(label="", markerstrokecolor=:auto, markersize=6, legendfontsize=12, guidefontsize=13, tickfontsize=10)
using LaTeXStrings
using MIRTjim: prompt
using InteractiveUtils: versioninfo


# The following line is helpful when running this jl-file as a script;
# this way it will prompt user to hit a key after each image is displayed.

isinteractive() && prompt(:prompt);


# ## Under-determined case

A = [1 2] # 1st case: M < N
y = [3] # obviously x=[1 1] is one possible solution (but not MNLS)
f1(x) = norm(A*x - y)
xh1 = A \ y

x1 = LinRange(-1,1,101) * 3
x2 = LinRange(-1,1,103) * 3
c1 = [f1([x1a, x2a]) for x1a=x1, x2a=x2];

color = :viridis
contour(x1, x2, c1', label="contours"; color)
plot!(aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
scatter!([0], [0], color=:black, markershape=:square, label="")
plot!([0, xh1[1]], [0, xh1[2]], line=:magenta, label="")
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label="{x : y=Ax}")
scatter!([1], [1], color=:blue, markershape=:star5, label="[1,1]")
scatter!([xh1[1]], [xh1[2]], color=:red, markershape=:circle, label="MNLS",
 title = "Under-determined case")

#
prompt()

#src savefig("demo_ls_cost1a.pdf")


# ## Square but singular case

A = [1 2; 2 4] # 2nd case: M = N but singular A
y = [3, 6] # again [1,1] is a solution (but not MNLS solution)
f2(x) = norm(A*x - y)
xh2 = pinv(A) * y

c2 = [f2([x1a, x2a]) for x1a=x1, x2a=x2]
contour(x1, x2, c2', label="contours"; color)
plot!(aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
scatter!([0], [0], color=:black, markershape=:square, label="")
plot!([0, xh2[1]], [0, xh2[2]], line=:magenta, label="")
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label="{x : y[1]=A[1,:] x}")
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dash), label="{x : y[2]=A[2,:] x}")
scatter!([1], [1], color=:blue, markershape=:star5, label="[1,1]")
scatter!([xh2[1]], [xh2[2]], color=:red, markershape=:circle, label="MNLS",
 title = "Singular case")

#
prompt()

#src savefig("demo_ls_cost1b.pdf")


# ## Square non-singular case

A = [1 2; 1 3] # 3rd case: M = N with non-singular A
y = [3, 4] # now x=[1,1] is the unique solution (by design)
f3(x) = norm(A*x - y)
xh3 = A \ y

c3 = [f3([x1a, x2a]) for x1a=x1, x2a=x2]
contour(x1, x2, c3', label="contours"; color)
plot!(aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label="{x : y[1]=A[1,:] x}")
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dash), label="{x : y[2]=A[2,:] x}")
scatter!([xh3[1]], [xh3[2]], color=:red, markershape=:circle, label="LLS",
 title = "Non-singular case")

#
prompt()

#src savefig("demo_ls_cost1c.pdf")


# ## Typical over-determined case

A = [1 2; 1 -1; 2 1] # 4th case: M > N with (typical) inconsistent data
y = [3, 2, 1] # no consistent solution
f4(x) = norm(A*x - y)
xh4 = A \ y

c4 = [f4([x1a, x2a]) for x1a=x1, x2a=x2]
contour(x1, x2, c4', label="contours"; color)
plot!(aspect_ratio=:equal, xlabel=L"x_1", ylabel=L"x_2", legend=:bottomleft)
plot!(x1, (y[1] .- A[1,1]*x1)/A[1,2], line=(:blue,:dash), label="{x : y[1]=A[1,:] x}")
plot!(x1, (y[2] .- A[2,1]*x1)/A[2,2], line=(:green,:dash), label="{x : y[2]=A[2,:] x}")
plot!(x1, (y[3] .- A[3,1]*x1)/A[3,2], line=(:purple,:dash), label="{x : y[3]=A[3,:] x}")
plot!(xlim=(-3,3), ylim=(-3,3))
scatter!([xh4[1]], [xh4[2]], color=:red, markershape=:circle, label="LLS",
 title = "Over-determined case")

#
prompt()

#src savefig("demo_ls_cost1d.pdf")


include("../../../inc/reproduce.jl")
