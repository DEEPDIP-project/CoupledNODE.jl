using CoupledNODE
using Documenter

DocMeta.setdocmeta!(CoupledNODE, :DocTestSetup, :(using CoupledNODE); recursive=true)

makedocs(;
    modules=[CoupledNODE],
    authors="Pablo Rodríguez Sánchez <pablo.rodriguez.sanchez@gmail.com>, Luisa Orozco <l.orozco@esciencecenter.nl>, Simone Ciarella <s.ciarella@esciencecenter.nl>, Aron Jansen <a.p.jansen@esciencecenter.nl>",
    sitename="CoupledNODE.jl",
    format=Documenter.HTML(;
        canonical="https://DEEPDIP-project.github.io/CoupledNODE.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],)

deploydocs(;
    repo="github.com/DEEPDIP-project/CoupledNODE.jl",
    devbranch="main",
)