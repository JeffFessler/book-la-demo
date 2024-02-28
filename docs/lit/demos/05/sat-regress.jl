#=
# [Linear regression and SAT scores](@id sat-regress)

This example is just a way to provide some template code
for a homework problem on linear regression.

The data here comes from
the 2007 paper by L. M. Lesser
titled
[Critical Values and Transforming Data: Teaching Statistics with Social Justice](http://doi.org/10.1080/10691898.2007.11889454)
and is based on data collected
by the
[College Board](https://en.wikipedia.org/wiki/College_Board),
the organization that runs the
[SAT exam](https://en.wikipedia.org/wiki/SAT)
for high-school students.
This data includes average SAT Math scores
for 10 different family annual income brackets.
The homework problem uses this data
to explore the relationship between income and SAT scores
via linear regression.
=#

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
        "LinearAlgebra"
        "MIRTjim"
        "Plots"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

using Plots: default, gui, scatter, savefig
default(); default(label="", markerstrokecolor=:auto, widen=true, linewidth=2,
 markersize = 6, tickfontsize=14, labelfontsize = 18, legendfontsize=16)


#=
## Data

Normally we would read such data
from a data file
such as a `.csv` file
using
[`CSV.jl`](https://github.com/JuliaData/CSV.jl).
This data is small enough
to just paste here directly.
=#

data = [
 "Income Bracket (in \$1000s)" "0 – 10" "10 – 20" "20 – 30" "30 – 40" "40 – 50" "50 – 60" "60 – 70" "70 – 80" "80 – 100" "100+"
 "Math" 457 465 474 488 501 509 515 521 534 564
 "Verbal" 429 445 462 478 493 500 505 511 523 549
 "Writing" 427 440 454 470 483 490 496 502 514 543
];

math = Int.(data[2,2:end]) # math scores

#
income = [5:10:75; 90; 120] # middle of each range

#
scatter(income, math; label="Data", legend = :bottomright,
 xaxis = ("Family Annual Income (1000\$)",),
 yaxis = ("SAT Average Math Score", (425,575), 425:50:575),
)

#
