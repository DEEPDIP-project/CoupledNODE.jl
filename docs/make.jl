using SciMLModelCoupling
using Documenter

DocMeta.setdocmeta!(SciMLModelCoupling, :DocTestSetup, :(using SciMLModelCoupling); recursive=true)

makedocs(;
    modules=[SciMLModelCoupling],
    authors="Pablo Rodríguez Sánchez <pablo.rodriguez.sanchez@gmail.com> and contributors",
    sitename="SciMLModelCoupling.jl",
    format=Documenter.HTML(;
        canonical="https://pabrod.github.io/SciMLModelCoupling.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pabrod/SciMLModelCoupling.jl",
    devbranch="main",
)
