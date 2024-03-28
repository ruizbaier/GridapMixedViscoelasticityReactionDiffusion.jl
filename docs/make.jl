using GridapMixedViscoelasticityReactionDiffusion
using Documenter

DocMeta.setdocmeta!(GridapMixedViscoelasticityReactionDiffusion, :DocTestSetup, :(using GridapMixedViscoelasticityReactionDiffusion); recursive=true)

makedocs(;
    modules=[GridapMixedViscoelasticityReactionDiffusion],
    authors="Mathieu Barre <mathieu.barre@monash.edu>, Ricardo Ruiz Baier <ricardo.ruizbaier@monash.edu>",
    repo="https://github.com/gridap/GridapMixedViscoelasticityReactionDiffusion.jl/blob/{commit}{path}#{line}",
    sitename="GridapMixedViscoelasticityReactionDiffusion.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gridap.github.io/GridapMixedViscoelasticityReactionDiffusion.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/gridap/GridapMixedViscoelasticityReactionDiffusion.jl",
    devbranch="main",
)
