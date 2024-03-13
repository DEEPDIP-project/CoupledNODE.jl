# For developers

This folder contains an example function, for pedagogical purposes only. Eventually we will delete it.

Please note that the function:

1. Is defined in `./src/toy/toyexample.jl`
   - Is explicitly exported
   - The file can contain multiple functions
2. Is included via `./src/SciMLModelCoupling.jl`. This ships the function via `using SciMLModelCoupling`
3. Is tested at `./test/runtests.jl`
4. Is used in an example in `./examples/00-Toy.jl`