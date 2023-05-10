# logo.jl book

using Colors: RGB
using FileIO: save
using Random: seed!, randperm
seed!(0)

const julia_purple  = (0.584, 0.345, 0.698)
const julia_green   = (0.22, 0.596, 0.149)
const julia_red     = (0.796, 0.235, 0.2)
colors = [ julia_purple, julia_green, julia_red, ] # from Luxor.jl
colors = [RGB{Float32}(c...) for c in colors]
black = RGB{Float32}(0, 0, 0)

M,N = 20,16
pcolor = 0.8 # probability of a color

logo = fill(black, M, N)
ii = randperm(M*N)[1:Int(pcolor*M*N)]
tmp = rand(1:3, length(ii)) # too clustered, despite random?  need poisson disc?
logo[ii] = colors[tmp]

z = zeros(3)
z = [z; ones(9); z]
logo = kron(logo, z*z')
save("logo.png", logo)
