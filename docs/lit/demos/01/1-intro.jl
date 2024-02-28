#=
# [Tutorial: Julia Overview](@id tutor-1-intro)

Julia overview.

- 2018-08-11 Julia 0.7.0 Jeff Fessler (based on 2017 version by David Hong)
- 2019-01-20 Julia 1.0.3 and add note about line breaks
- 2020-08-05 Julia 1.5.0
- 2021-08-23 Julia 1.6.2
- 2023-09-03 Julia 1.9.2, Literate
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
        "InteractiveUtils"
        "LinearAlgebra"
    ])
end


# Tell Julia to use the following packages.
# Run `Pkg.add()` in the preceding code block first, if needed.

## using InteractiveUtils: versioninfo
using LinearAlgebra: Diagonal, det, dot, tr


#=
## Numbers, arithmetic, types
=#

# Define a real number:
r = 3.0

# Variables in Julia have a type:
typeof(r)

#
i = 3

#
typeof(i)

#
c = 3. + 2im

#
typeof(c)

# We can add, subtract, multiply and divide like usual:
4. + 5

#
4. - 5

#
4. * 3

#
2. / 3

# Dividing `Int` values with `/` produces a `Float`:
2/3

#
4/2

# This is different from Python 2, but similar to Python 3.

# To divide integers with rounding, use `÷` instead.
# Type `\div` then hit tab:

5 ÷ 2

#=
More info about numbers here:
- [numbers](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers)
- [operations](https://docs.julialang.org/en/v1/manual/mathematical-operations)
- [complex](https://docs.julialang.org/en/v1/manual/complex-and-rational-numbers)

## Vectors and matrices (i.e., arrays)

Make a vector of real numbers:
```math
x = \begin{bmatrix} 1.0 \\ 3.5 \\ 2 \end{bmatrix}
```
=#

x = [1, 3.5, 2]

#=
Note the type: `Vector{Float64}`.

Having just one real number in the array sufficed
for the array have all `Float64` elements.

This is a true **one**-dimensional array of **`Float64`** values.

(Matlab does not have 1D arrays; it fakes it using 2D arrays of size `N × 1`.)
=#

size(x) # returns a tuple

#
length(x)

#
x_ints = [1,3,2]

# This is a **one**-dimensional array of **`Int64`** values.
# We use these less often.

size(x_ints)

#
length(x_ints)

#=
# Make a matrix using a semicolon to separate rows:

```math
A = \begin{bmatrix}
1.1 & 1.2 & 1.3 \\
2.1 & 2.2 & 2.3
\end{bmatrix}
```
=#

A = [1.1 1.2 1.3; 2.1 2.2 2.3]

# This is a **two**-dimensional array (aka a matrix) of **`Float64`** values.

size(A)

#
length(A)

# Different from Matlab, `length` always returns the total number of elements.

# Make vectors and matrices of all zeros.

zeros(3)

# Different from Matlab!
# Do not write `zeros(3,1)` because Julia has proper 1D arrays.
# `zeros(3,1)` and `zeros(3)` are different!

zeros(2,3)

# And ones:

ones(3)

#
ones(2,3)

#=
The "identity matrix" ``I``
in Julia's `LinearAlgebra` package is sophisticated.

Look at the following examples:
=#

using LinearAlgebra: I
ones(3,3) - I

#
ones(2,2) * I

#
I(3)

#=
If that ``I`` seems too fancy, then you could make your own `eye` function
akin to Matlab as follows
(but it should not be needed and it uses unnecessary memory):
=#

eye = n -> Matrix(1.0*I(n))
eye(2)

# Make diagonal matrices using the `Diagonal` function in `LinearAlgebra`:

Diagonal(3:6)

#=
This is far more memory efficient than Matlab's `diag` command
or Julia's `LinearAlgebra.diagm` method.  Avoid using those!
=#

#=
Make random vectors and matrices.

```math
x = \begin{bmatrix}
 \mathcal{N}(0,1) \\ \mathcal{N}(0,1) \\ \mathcal{N}(0,1)
 \end{bmatrix}
\qquad \text{i.e.,}
\quad x_i \overset{\text{iid}}{\sim} \mathcal{N}(0,1)
```

```math
A = \begin{bmatrix}
\mathcal{N}(0,1) & \mathcal{N}(0,1) & \mathcal{N}(0,1) \\
\mathcal{N}(0,1) & \mathcal{N}(0,1) & \mathcal{N}(0,1)
\end{bmatrix}
\qquad \text{i.e., }
\quad A_{ij} \overset{\text{iid}}{\sim} \mathcal{N}(0,1)
```
=#

x = randn(3)

#
A = randn(2,3)

#=
## Matrix operations

Indexing is done with **square** brackets
(like in C and Python, unlike Matlab).

Index and begins at **1**
(like in Matlab and counting)
not **0** (like in C or Python).
=#

A = [1.1 1.2 1.3; 2.1 2.2 2.3]

#
A[1,1]

#
A[1,2:3]

# This row-slice is a one-dimensional slice (!) not a `1×2` matrix:

A[1:2,1]

#
A[2,:]

# Vector dot product:

x = randn(3)
xdx = x'x

#
xdx = dot(x,x)

#
xdx = x'*x

# Different from Matlab! The output is a scalar, **not** a `1×1` "matrix:"

typeof(xdx)

# Matrix times vector:

A = randn(2,3)
x = randn(3)
A*x

# Matrix times matrix:

A = randn(2,3)
B = randn(3,4)
A*B

# Matrix transpose (conjugate and non-conjugate):

A = 10*reshape(1:6, 2, 3) + im * reshape(1:6, 2, 3)

# conjugate transpose, could also use `adjoint(A)`:
A'

#=
For complex arrays, rarely do we need a non-conjugate transpose.
Usually we need `A'` instead.  But if we do:
=#

transpose(A) # essentially sets a flag about transpose without reordering data

# Matrix determinant:

A = Diagonal(2:4)
det(A)

#
B = randn(3,3)
[det(A*B) det(A)*det(B)]

# Matrix trace:

A = ones(3,3)
tr(A) # in Matlab would be "trace(A)"

#=
More info in
[Julia manual](https://docs.julialang.org/en/v1/manual/arrays)

## Getting help

Julia analogue of Matlab's `help` is `?`.

Type `?pwd` in the REPL to get help on the `pwd` function.

It does not work in this online documentation so we use `@doc` instead:
=#

@doc pwd

#=
- [Full documentation](https://docs.julialang.org)
- Searching Julia's
  [Github repo](https://github.com/JuliaLang/julia)
  can sometimes also uncover similar issues.
- Lots of neat talks on their
  [YouTube channel](https://www.youtube.com/user/JuliaLanguage)
- Here is an interesting one about
  [vector transpose](https://www.youtube.com/watch?v=C2RO34b_oPM)

## Ranges
Ranges are different from (and much more efficient than) Matlab!
=#

myrange = -2:3

#
typeof(myrange)

# Not an Array! But it can be indexed:

myrange[1]

# Used often in `for` loops:

for a in myrange
    println(a)
end

# Form an array by using `collect` if needed (use rarely):

collect(myrange)

# Other ways to make ranges:

srange = 1:-1:-5

#
typeof(srange)

#
lrange = range(0, 2, 10)

#
typeof(lrange)

# Yet another option that looks the most like `linspace`:
LinRange(0,10,6)

#=
## Comprehensions
=#

# A convenient way to create arrays!

comp = [i+0.1 for i in 1:5]

#
comp = [10i + j for i in 1:5, j in 1:4]

# ## Defining functions

# Way 1:

function f1(x,y)
    z = x+y
    return z
end

# Way 2:

f2(x,y) = x+y

# Way 3: Anonymous function:

f3 = (x,y) -> x+y

# Functions can return multiple outputs:

function f_mult(x, y)
    add = x + y
    sub = x - y
    return add, sub
end;

f_mult(2,3)

# The output is a `Tuple` of the returned values:
out_tuple = f_mult(2,3)

#
typeof(out_tuple)


# Convenient way to split out the outputs:

out1, out2 = f_mult(2,3)

#
out1

#
out2

#=
## Broadcast

Any Julia function can be "vectorized" using "broadcast"
=#

myquad = x -> (x+1)^2

#
myquad(1)

#
try
    myquad([1,2,3]) # this does not work!
catch
    "failed, as expected"
end

#=
This particular function was not designed
to be applied to vector input arguments!
But it can be used with vectors (or arrays)
by adding a `.` to tell Julia to apply it element-wise.
This is called
[broadcasting](https://docs.julialang.org/en/v1/manual/arrays/#Broadcasting).
=#

myquad.([1,2,3])


#=
## Conditionals

`if` `else` `end` `for`

Generally similar to Matlab.
Optional use of `in` instead of `=` in the for loop.
=#

for j in 1:3
    if j == 2
        println("$j is a two! ^^")
    else
        println("$j is not a two. :(")
    end
end

# Julia has the convenient ternary operator:

mystring = 2 > 3 ? "2 is greater than 3" : "2 is not greater than 3"


#=
## Plotting
Suggested package: `Plots.jl` with its default `gr` backend.

**Note:** Usually slower the first time you plot due to precompiling.
You must `add` the "Plots" package first.
In a regular Julia REPL you do this
by using the `]` key to enter the package manager REPL,
and then type `add Plots` then wait.

In a Jupyter notebook,
type `using Pkg` then `add Plots` and wait.
=#

using Plots
backend()

# Plot values from a vector.  (The labels are optional arguments.)

x = range(-5,5,101)
y = x.^2
plot(x, y, xlabel="x", ylabel="y", label="parabola")

#=
## `heatmap`
=#

x = range(-2, 2, 101)
y = range(-1.1, 1.1, 103)
A = x.^2 .+ 30 * (y.^2)'
F = exp.(-A)
p1 = heatmap(x, y, F', # for F(x,y)
    color=:grays, aspect_ratio=:equal, xlabel="x", ylabel="y", title="bump")

#=
Using the `jim` function the `MIRTjim.jl` package simplifies
the display of 2D images, among other features.
See its
[examples](https://jefffessler.github.io/MIRTjim.jl/stable/generated/examples/1-examples).
=#


#=
## Plotting functions

`Plots.jl` allows you to pass in the domain and a function.
It does the rest. :)
This is one many examples of how Julia exploits "multiple dispatch."
=#

plot(range(0,1,100), abs2, label="x^2")

#
heatmap(range(-2,2,102), range(-1.1,1.1,100),
    (x,y) -> exp(-x^2-30*y^2), aspect_ratio=1)

#=
More info about plotting at
[https://juliaplots.github.io](https://juliaplots.github.io)
=#


#=
## Caution: line breaks (newlines)

If you want an expression to span multiple lines,
then be sure to enclose it in parentheses.
=#

# Compare the following 3 (actually 4) expressions:
x = 9
    - 7

#
y = 9 -
    7

#
z = (9
    - 7)

#
(x,y,z)


#=
## Submitting homework

This part is just for EECS 551 students at UM.

A quick example to try submitting problems.

**Task:**
Implement a function that takes two inputs and outputs them in reverse order.
=#

function template1(x, y)
    return (y, x)
end;

template1(2, 3)

#=
Copy the above function code into a file named `template1.jl`
and email to `eecs551@autograder.eecs.umich.edu`.

Make sure that:
- All reasonable input types can be handled.
  Internally trying to convert a `Float64` to an `Int64` can produce `InexactError`
- File extension is `.jl`. Watch out for hidden extensions!
- File has just the Julia function.
- (Your HW solutions can also contain `using` statements.)

An undocumented function is bad programming practice.
Julia supports `docstrings` for comments like this:
=#

"""
    template2(x,y)
This function reverses the order of the two input arguments.
"""
function template2(x,y)
    return (y,x)
end

# You can see the docstring by using the `?` key or `@doc`:

@doc template2
