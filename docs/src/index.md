
# Demos for Book

## Overview

[https://github.com/JeffFessler/book-la-demo](https://github.com/JeffFessler/book-la-demo)

Demos
in the
[Julia language](https://julialang.org),
compiled using
[Literate](https://github.com/fredrikekre/Literate.jl)
and
[Documenter](https://github.com/JuliaDocs/Documenter.jl)
to accompany the 2024 book
[Linear Algebra for Data Science, Machine Learning, and Signal Processing](https://www.cambridge.org/highereducation/books/linear-algebra-for-data-science-machine-learning-and-signal-processing/1D558680AF26ED577DBD9C4B5F1D0FED#overview)
by Jeff Fessler
and Raj Nadakuditi
at the University of Michigan.


## Getting started with Julia

* Install Julia from
  [https://julialang.org](https://julialang.org/downloads)

* Launch the Julia app;
  it should open a
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
  you should see a ☰
  [hamburger menu button](https://en.wikipedia.org/wiki/Hamburger_button)
  that will toggle open the demos menu sidebar.

* Check out some [Julia tutorials](https://julialang.org/learning/tutorials),
  especially the one titled
  "Just the Julia you need to get started in Data Science and ML" by Raj Rao.


## More resources

* Install the
  [web browser shortcut](https://discourse.julialang.org/t/how-to-search-the-manual-more-efficiently/19314)
  for fast access to the online
  [Julia manual](https://docs.julialang.org/en/v1).

* Use the package
  [AbbreviatedStackTraces.jl](https://github.com/BioTurboNick/AbbreviatedStackTraces.jl)
  to get more interpretable error messages.

* For image processing,
  view the excellent documentation at
  [JuliaImages](https://juliaimages.org)

* For a machine learning introduction, see the
  [Julia programming for Machine Learning course material](https://github.com/adrhill/julia-ml-course).


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
[SVD demo](https://jefffessler.github.io/book-la-demo/generated/demos/04/svd-diff)
is at
[this url](https://github.com/JeffFessler/book-la-demo/blob/main/docs/lit/demos/04/svd-diff.jl).
After downloading such a file such as
[svd-demo.jl](https://github.com/JeffFessler/book-la-demo/blob/main/docs/lit/demos/04/svd-diff.jl),
you can run it
by typing
`include("svd-demo.jl")`
at the Julia REPL.
