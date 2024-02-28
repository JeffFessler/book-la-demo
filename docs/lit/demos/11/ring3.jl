#=
# [Classification with MLP](@id ring3)

This demo
illustrates basic artificial NN training
for a simple synthetic classification example
with cross-entropy loss
using Julia's `Flux` library.

- Jeff Fessler, University of Michigan
- 2018-10-18 Julia 1.0.1 original
- 2024-02-26 Julia 1.10.1 update
=#


#srcURL


# ## Setup

# Packages needed here.

import Flux # Julia package for deep learning
using Flux: Dense, Chain, relu, params, Adam, throttle, mse
using Flux.Losses: logitcrossentropy
using OneHotArrays: onehotbatch
using InteractiveUtils: versioninfo
using LaTeXStrings # pretty plot labels
using MIRTjim: jim, prompt
using Random: seed!, randperm
using Plots: Plot, plot, plot!, scatter!, default, gui, savefig
using Plots.PlotMeasures: px

default(markersize=5, markerstrokecolor=:auto, label="",
 legendfontsize=16, labelfontsize=16, tickfontsize=14)

# The following line is helpful when running this file as a script;
# this way it will prompt user to hit a key after each figure is displayed.

isinteractive() ? jim(:prompt, true) : prompt(:draw);

#=
## Generate (synthetic) data

This data is suitable for a "hand crafted" classifier.

(The `Xvm` versions are vectors of matrices.)
=#

# Functions to simulate data that cannot be linearly separated
function sim_ring(
    n::Int, # number of points
    r::Real, # radius of center of annulus center
    σ::Real, # spread (annulus width)
)
    T = promote_type(eltype(r), eltype(σ))
    rad = r .+ σ * randn(T, n)
    ang = T(2π) * rand(T, n)
    return Matrix([rad .* cos.(ang)  rad .* sin.(ang)]') # (2,n)
end;

K = 3 # classes
nsim = (40, 80, 120) # how many in each class
rsim = (0, 3, 6) # mean radii of each class
dsim = (1.5, 4.5) # ideal decision boundaries
function simdata(;
    n = nsim,
    r = rsim,
    σ = Float32.((0.7, 0.5, 0.5)),
)
    Xvm = [sim_ring(n[k], r[k], σ[k]) for k in 1:K] # [K][n]
    Yvm = [fill(k, 1, n[k]) for k in 1:K] # [K][n]
    return (Xvm, Yvm)
end;


# Scatter plot function
function plot_data!(p, Xvm;
    colors = (:blue, :red, :orange),
    marks = (:circle, :star, :uptri),
)
    for k in 1:K
        scatter!(p, Xvm[k][1,:], Xvm[k][2,:],
           marker=marks[k], color=colors[k], label="class $k")
    end
    tmp = range(0, 2π, 101)
    for d in dsim
        plot!(p, d * cos.(tmp), d * sin.(tmp), color=:gray)
    end
    return p
end

function plot_data(Xvm; kwargs...)
    p = plot(
     xaxis = (L"x_1", (-1,1).*8, -9:3:9),
     yaxis = (L"x_2", (-1,1).*8, -9:3:9),
     aspect_ratio = 1, size = (500,500);
     kwargs...,
    )
    plot_data!(p, Xvm)
end;


# Training data
seed!(0)
ntrain = nsim
(Xvm_train, Yvm_train) = simdata(; n=ntrain)
p0 = plot_data(Xvm_train)
## savefig(p0, "ring3-data.pdf")

#
prompt()


# Validation and testing data
(Xvm_valid, Yvm_valid) = simdata()
(Xvm_test, Yvm_test) = simdata()

p1 = plot_data(Xvm_valid; title="Validation")
p2 = plot_data(Xvm_test; title="Test")

p0t = plot!(deepcopy(p0); title="Train")
p3 = plot(p0t, p1, p2;
 leftmargin = 30px, bottommargin = 35px,
 size = (1600,500), layout=(1,3),
)

#
prompt()


#=
## Hand-crafted classifier
This data is not linearly separable,
but a simple nonlinearity makes it so.
=#
lift1 = x -> [x; sqrt(sum(abs2, x))] # lift 1 feature vector
lifter = xx -> mapslices(lift1, xx, dims=1) # apply to each data column
Xvm_train_lift = lifter.(Xvm_train)
Xm_train_lift = hcat(Xvm_train_lift...)
pl_train = plot(Xm_train_lift[3,:], marker=:circle, yticks=(1:5)*1.5,
 xticks = cumsum([1; collect(ntrain)]),
)

#
prompt()

#=
So the levels 1.5 and 4.5 should work to separate classes.
Try on the test data.
=#
Xm_test_lift = hcat(lifter.(Xvm_test)...)
lift_classifier = x -> x < dsim[1] ? 1 : x > dsim[2] ? 3 : 2
class_test_lift = lift_classifier.(Xm_test_lift[3,:])
test_lift_errors = count(class_test_lift .!= hcat(Yvm_test...)')
@show test_lift_errors, size(Xm_test_lift, 2)


#=
Permute the training data (just to be cautious)
=#
Xm_train = hcat(Xvm_train...)
Ym_train = hcat(Yvm_train...)
perm = randperm(size(Xm_train,2))
Xm_train = Xm_train[:,perm] # (2, sum(nsim))
Ym_train = Ym_train[:,perm] # (1, sum(nsim))
Xm_valid = hcat(Xvm_valid...)
Ym_valid = hcat(Yvm_valid...)
Xm_test = hcat(Xvm_test...)
Ym_test = hcat(Yvm_test...);


#=
## Train simple MLP model

A
[multilayer perceptron model]
(https://en.wikipedia.org/wiki/Multilayer_perceptron)
(MLP)
consists of multiple fully connected layers.

Train a basic NN model with 1 hidden layer;
here using MSE loss just for illustration.
=#

if !@isdefined(state1)
    nhidden = 10 # neurons in hidden layer
    model = Chain(Dense(2, nhidden, relu), Dense(nhidden, 1))
    loss3ms(model, x, y) = mse(model(x), y) # admittedly silly choice
    iters = 2000
    dataset = Base.Iterators.repeated((Xm_train, Ym_train), iters)
    state1 = Flux.setup(Adam(), model)
    Flux.train!(loss3ms, model, dataset, state1)
end;

scalar = y -> y[1]
model1 = x -> scalar(model(x))


# Plot results after training

function display_decision_boundaries(
    model;
    x1range = range(-1f0,1f0,101)*8,
    x2range = x1range,
    kwargs...,
)
    D = [model1([x1;x2]) for x1 in x1range, x2 in x2range]
    D = round.(D)
    jim(x1range, x2range, D; color=:cividis,
     xaxis = (L"x_1", (-1,1).*8, -9:3:9),
     yaxis = (L"x_2", (-1,1).*8, -9:3:9),
     aspect_ratio = 1, size = (500,500),
    kwargs...)
end;

function display_decision_boundaries(model, Xvm; kwargs...)
    p = display_decision_boundaries(model; kwargs...)
    plot_data!(p, Xvm)
end


# Examine classification accuracy
classacc(model, x, y::Number) = round(model1(x)) == y
classacc(model, x, y::AbstractArray) = classacc(model, x, y[1])
function classacc(Xm, Ym)
    tmp = zip(eachcol(Xm), eachcol(Ym))
    tmp = count(xy -> classacc(model, xy...), tmp)
    tmp = tmp / size(Ym,2) * 100
    return round(tmp, digits=3)
end

lossXYtrain = loss3ms(model, Xm_train, Ym_train)
p4 = display_decision_boundaries(model, Xvm_train;
 title = "Train: RMSE Loss = $(round(sqrt(lossXYtrain),digits=4)), " *
 "Class=$(classacc(Xm_train, Ym_train)) %");
lossXYtest = loss3ms(model, Xm_test, Ym_test)
p5 = display_decision_boundaries(model, Xvm_test;
     title = "Test: RMSE Loss = $(round(sqrt(lossXYtest),digits=4)), " *
    "Class=$(classacc(Xm_test, Ym_test)) %");
plot(p4, p5; size=(1000,500))

#
prompt()


#=
## Train while validating

This time using cross-entropy loss,
which makes more sense for a classification problem.

Create a basic NN model with 1 hidden layer.
This version evaluates performance every epoch
for both the training data and validation data.
=#

if !@isdefined(model2) # || true
    layer2 = Dense(2, nhidden, relu)
    layer3 = Dense(nhidden, K)
    model2 = Chain(layer2, layer3)
    onehot1 = y -> reshape(onehotbatch(y, 1:K), K, :)
    loss3ce(model, x, y) = logitcrossentropy(model(x), onehot1(y))

    nouter = 80 # of outer iterations, for showing loss
    losstrain = zeros(nouter+1)
    lossvalid = zeros(nouter+1)

    iters = 100 # inner iterations
    losstrain[1] = loss3ce(model2, Xm_train, Ym_train)
    lossvalid[1] = loss3ce(model2, Xm_valid, Ym_valid)

    model2s = similar(Vector{Any}, nouter) # to checkpoint every outer iteration
    for io in 1:nouter
        ## @show io
        idataset = Base.Iterators.repeated((Xm_train, Ym_train), iters)
        istate = Flux.setup(Adam(), model2)
        Flux.train!(loss3ce, model2, idataset, istate)
        losstrain[io+1] = loss3ce(model2, Xm_train, Ym_train)
        lossvalid[io+1] = loss3ce(model2, Xm_valid, Ym_valid)
        if (io ≤ 6) && false # set to true to make images
            display_decision_boundaries(model2, Xvm_train)
            plot!(title="$(io*iters) epochs")
            gui(); sleep(0.3)
        end
        model2s[io] = deepcopy(model2)
    end
end;


# Show loss vs epoch
ivalid = findfirst(>(0), diff(lossvalid))
#src ivalid = isnothing(ivalid) ? 0 : ivalid
plot(xlabel="epoch/$(iters)", yaxis=("CE loss", (0,1.05*maximum(losstrain))))
plot!(0:nouter, lossvalid, label="validation", marker=:+, color=:violet)
plot!(0:nouter, losstrain, label="training", marker=:o, color=:green)
plot!(xticks = [0, ivalid, nouter])
#src savefig("ml-flux-loss.pdf")

#
prompt()

model2v = model2s[ivalid]; # model at validation epoch
lossf = (Xm,Ym) -> round(loss3ce(model2v, Xm, Ym), digits=4)
loss_train = lossf(Xm_train, Ym_train)
loss_valid = lossf(Xm_valid, Ym_valid)
loss_test = lossf(Xm_test, Ym_test);

p1 = display_decision_boundaries(model2v, Xvm_train;
 title="Train:\nCE Loss = $loss_train\n" *
    "Class=$(classacc(Xm_train, Ym_train)) %",
)
p2 = display_decision_boundaries(model2v, Xvm_valid;
 title="Valid:\nCE Loss = $loss_valid\n" *
    "Class=$(classacc(Xm_valid, Ym_valid)) %",
)
p3 = display_decision_boundaries(model2v, Xvm_test;
 title="Test:\nCE Loss = $loss_test\n" *
    "Class=$(classacc(Xm_valid, Ym_valid)) %",
)
p123 = plot(p1, p2, p3; size=(1500,500), layout=(1,3))

#
prompt()

# Show response of (trained) first hidden layer, at validation step
x1range = range(-1f0,1f0,31) * 6
x2range = range(-1f0,1f0,33) * 6
tmp = model2v.layers[1]
layer2data = [tmp([x1;x2])[n] for x1 = x1range, x2 = x2range, n in 1:nhidden]

pl = Array{Plot}(undef, nhidden)
for n in 1:nhidden
    ptmp = jim(x1range, x2range, layer2data[:,:,n], color=:cividis,
        xtick=-6:6:6, ytick=-6:6:6,
    )
    if n == 7
        plot!(ptmp, xlabel=L"x_1", ylabel=L"x_2")
    end
    pl[n] = ptmp
end
plot(pl[1:9]...)

#
prompt()


# ### Reproducibility

# This page was generated with the following version of Julia:
io = IOBuffer(); versioninfo(io); split(String(take!(io)), '\n')

# And with the following package versions
import Pkg; Pkg.status()
