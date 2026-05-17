Errata
for the
[2024 Cambridge book](https://web.eecs.umich.edu/~fessler/#:~:text=https%3A//www.cambridge.org/highereducation/isbn/9781009418140)
"Linear Algebra for Data Science, Machine Learning, and Signal Processing"
by Jeff Fessler and Raj Nadakuditi of the University of Michigan.

We are grateful to
Rodrigo Lobos
for finding many of these errors.
Other contributors:
Laura Balzano, Amaya Murguia, Jacob Root.

p42 The remark about left division `\(A,I)` is outdated.
For `inv`,
Julia 1.10 uses `lu` that calls `LAPACK.getrf!` for LU decomposition.

p72
If …(V’)… represents counter-clockwise…, then …(V)… must represent clockwise…
Here, counter-clockwise and clockwise are swapped.

p109 Definition of row space should be ℛ(A^⊤) instead of ℛ(A')

p112 Fact 4.20: 𝐅 should be 𝔽

p113 Eqn. (4.27): 𝐅 should be 𝔽

p199 Q6.1 answer E should be 6.2a,c,d not a,c,c

p215 Eqn. (6.72): 𝒫_n should be 𝒫_N

p216 Eqn. (6.73): 𝒫_n should be 𝒫_K

p216 Eqn. (6.73): k(n) should be p(k)

p218 First line of section 6.4.7, the norm should be squared

p224 Eqn. (6.97) end of middle line: trace{B A' ^Q} should be trace{B A' ^Q'}

p228 Example 6.21: B₁'B₂ should be 1/√2 [1 1; 1 -1]

p246 Frobenious should be Frobenius

p265 Eqn. (7.40): β* should be defined as minimizer of SURE(β).
Then by (7.39) this is approximately the minimizer of the MSE.

p299 Eqn. (8.38) upper limit of union should be "N" not "n"

p301 nonnnegative should be nonnegative

p306 "irreducibly diagonally dominant" should have been
[defined](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix#Variations).

p354 Eqn. (9.44) is missing the "3" in "3/32" in the numerator

p356 In Q9.24 N=2 (clearly)

p360 In Example 9.17 and Fig. 9.9, N=2 should be N=6.
See [the demo](https://jefffessler.github.io/book-la-demo/generated/demos/09/logistic1).

p399 Eqn. (12.26): the first σ₊ should be σ₋ instead.

p408 Ref. [65] by Lobos should be dated Jan. 2024
 [doi 10.1109/TMI.2023.3297851](https://doi.org/10.1109/TMI.2023.3297851)

p415 Ref. [212] by Lipor should be dated Mar. 2021
 [doi 10.1093/imaiai/iaab026](https://doi.org/10.1093/imaiai/iaab026)
