# logo.jl book

using Colors: RGB
using ColorVectorSpace # for math
using FileIO: save
using Random: seed!, randperm
seed!(0)

const julia_purple  = (0.584, 0.345, 0.698)
const julia_green   = (0.22, 0.596, 0.149)
const julia_red     = (0.796, 0.235, 0.2)
color1 = [julia_purple, julia_green, julia_red] # from Luxor.jl
colors = [RGB{Float32}(c...) for c in color1]
# black = RGB{Float32}(0, 0, 0)

# 300 dpi for 8x10 means 2400x3000 = 16*150 x 20*150
K = 150 # box size for "big" 300dpi version
M,N = 20,16 # grid size 20/16 = 10/8

function chiclet(c; pad::Float32=1.2f0)
    u = range(-1, 1, K) * pad
    p = 8
    tmp = @. (u^p .+ u'^p)^(1/p)
    f(x) = x < 0.97 ? 1 : x > 0.99 ? 0 : (0.99 - x) / 0.02
    tmp = @. (u^p .+ u'^p)^(1/p)
    mask = f.(tmp)
    x = range(-1, 1, K) * 0.09
    tmp = [RGB{Float32}((c .- (1,1,1) .* x)...) for x in x]
    tmp = repeat(tmp, 1, K)
    tmp .*= mask
    return tmp
end
boxes = chiclet.(color1)

pcolor = 0.8 # probability of a color

inds = zeros(Int, M, N)
ii = randperm(M*N)[1:floor(Int,pcolor*M*N)]
inds[ii] = rand(1:3, length(ii)) # too clustered?  need poisson disc?

logo = kron(inds .== 1, boxes[1])
for i in 2:3
    logo .+= kron(inds .== i, boxes[i])
end
# save("logo-big.png", logo); run(`xv logo-big.png`)
# save("logo.png", logo): run(`xv logo.png`)
