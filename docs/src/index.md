
# Demos for Book

## Overview

[https://github.com/JeffFessler/book-mmaj-demo](https://github.com/JeffFessler/book-mmaj-demo)

Demos
in the
[Julia language](https://julialang.org),
compiled using
[Literate](https://github.com/fredrikekre/Literate.jl)
and
[Documenter](https://github.com/JuliaDocs/Documenter.jl)
to accompany the book
"Matrix Methods and Applications in Julia"
by Jeff Fessler
and Raj Nadakuditi
at the University of Michigan.


## Getting started with Julia

* Install Julia from
  [https://julialang.org](https://julialang.org/downloads)

* Launch the Julia app
  should open a
  [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL).

* To develop code, select an editor, preferably with Julia integration, such as
  [VSCode](https://www.julia-vscode.org)
  or [vim](https://github.com/JuliaEditorSupport/julia-vim)
  perhaps with
  [tmux](https://discourse.julialang.org/t/julia-vim-tutorial-for-newbies/36636).
  Appropriate
  [editor plug-ins](https://github.com/JuliaEditorSupport)
  are needed to use LaTeX-like
  [tab-completion](https://docs.julialang.org/en/v1/stdlib/REPL/#Tab-completion)
  of
  [unicode](https://docs.julialang.org/en/v1/manual/unicode-input/#Unicode-Input)
  characters like `÷ ⊗ ⊕ ∘ ×` and `α β γ`.

* Peruse the demos listed in the menu here.
  If your browser window is wide enough,
  you should see a menu to the left.
  If your window is narrow,
  you should see a
  [hamburger menu button](https://en.wikipedia.org/wiki/Hamburger_button)
  that will toggle open the demos menu sidebar.

* View the excellent documentation at
  [JuliaImages](https://juliaimages.org)

* Check out some [Julia tutorials](https://julialang.org/learning/tutorials),
  especially the one titled
  "Just the Julia you need to get started in Data Science and ML" by Raj Rao.


## Getting started with Julia for matrix methods

These examples show you Julia code
and the corresponding output
in an HTML format suitable for viewing
in a web browser
without installing *any* software.

You could cut and paste portions of that Julia code
into the
[Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL),
but that becomes tedious.
Instead,
click on the "Edit on GitHub" link
(in the upper right, with
[github icon](https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png)),
where you can then download the entire Julia code file
that generated any of these examples.

For example,
the code for the
[SVD demo](https://jefffessler.github.io/book-mmaj-demo/generated/demos/03/svd-diff)
is at
[this url](https://github.com/JeffFessler/book-mmaj-demo/blob/main/docs/lit/demos/03/svd-diff.jl).
After downloading such a file such as
[svd-demo.jl](https://github.com/JeffFessler/book-mmaj-demo/blob/main/docs/lit/demos/03/svd-diff.jl),
you can run it
by typing
`include("svd-demo.jl")`
at the Julia REPL.
