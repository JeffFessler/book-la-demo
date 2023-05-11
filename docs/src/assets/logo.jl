# logo.jl book

using Colors: RGB
using FileIO: save
using Random: seed!, randperm
seed!(0)

const julia_purple  = (0.584, 0.345, 0.698)
const julia_green   = (0.22, 0.596, 0.149)
const julia_red     = (0.796, 0.235, 0.2)
color1 = [julia_purple, julia_green, julia_red] # from Luxor.jl
colors = [RGB{Float32}(c...) for c in color1]
# black = RGB{Float32}(0, 0, 0)

N = 50 # box size
function chiclet(c; pad::Float32=1.2f0)
    u = range(-1, 1, N) * pad
    p = 8
    mask = @. (u^p .+ u'^p)^(1/p) < 0.99
    x = range(-1, 1, N) * 0.09
    tmp = [RGB{Float32}((c .- (1,1,1) .* x)...) for x in x]
    tmp = tmp * ones(N)' .* mask
    return tmp
end
boxes = chiclet.(color1)

M,N = 20,16 # grid size
pcolor = 0.8 # probability of a color

inds = zeros(Int, M, N)
ii = randperm(M*N)[1:Int(pcolor*M*N)]
inds[ii] = rand(1:3, length(ii)) # too clustered?  need poisson disc?

logo = kron(inds .== 1, boxes[1])
for i in 2:3
    logo .+= kron(inds .== i, boxes[i])
end
# save("logo.png", logo)
# run(`xv logo.png`)
