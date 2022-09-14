execute = isempty(ARGS) || ARGS[1] == "run"

org, reps = :JeffFessler, "book-mmaj-demo"
using Documenter
using Literate

# https://juliadocs.github.io/Documenter.jl/stable/man/syntax/#@example-block
ENV["GKSwstype"] = "100"
ENV["GKS_ENCODING"] = "utf-8"

# generate examples using Literate
lit = joinpath(@__DIR__, "lit")
src = joinpath(@__DIR__, "src")
gen = joinpath(@__DIR__, "src/generated")

base = "$org/$reps"
repo_root_url =
    "https://github.com/$base/blob/main/docs/lit/demos"
nbviewer_root_url =
    "https://nbviewer.org/github/$base/tree/gh-pages/generated/demos"
binder_root_url =
    "https://mybinder.org/v2/gh/$base/gh-pages?filepath=generated/demos"

for (root, _, files) in walkdir(lit), file in files
    splitext(file)[2] == ".jl" || continue # process .jl files only
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit => gen))[1]
    Literate.markdown(ipath, opath, documenter = execute; # run examples
        repo_root_url, nbviewer_root_url, binder_root_url)
    Literate.notebook(ipath, opath; execute = false, # no-run notebooks
        repo_root_url, nbviewer_root_url, binder_root_url)
end


# Documentation structure
ismd(f) = splitext(f)[2] == ".md"
pages(folder) =
    [joinpath("generated/", folder, f) for f in readdir(joinpath(gen, folder)) if ismd(f)]
demos(folder) = pages(joinpath("demos", folder))

isci = get(ENV, "CI", nothing) == "true"

format = Documenter.HTML(;
    prettyurls = isci,
    edit_link = "main",
    canonical = "https://$org.github.io/$reps/stable/",
    assets = ["assets/custom.css"],
)

makedocs(;
    modules = Module[],
    authors = "Jeff Fessler and contributors",
    sitename = "Demos",
    format,
    pages = [
        "Home" => "index.md",
#       "00 Matrix" => demos("00"),
        "01 Matrix" => demos("01"),
#       "02 Eig/SVD" => demos("02"),
        "03 Subspaces" => demos("03"),
        "04 LS" => demos("04"),
        "05 Norm" => demos("05"),
        "06 Low-Rank" => demos("06"),
#       "07 Special" => demos("07"),
#       "08 Optimize" => demos("08"),
#       "09 Complete" => demos("09"),
    ],
)

if isci
    deploydocs(;
        repo = "github.com/$base",
        devbranch = "main",
        devurl = "dev",
        versions = nothing,
        forcepush = true,
        push_preview = true,
    )
end
