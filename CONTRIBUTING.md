## Pre-commit
We use pre-commit to run some checks before you commit your changes.

To install it follow instructions [here](https://pre-commit.com/) and then run:

```bash
pre-commit install
```

in the root of the repository.

Now every time you commit your changes, pre-commit will run a formatting check and create any notebooks whose source files have changed. 
If it made any changes, you'll have to manually stage them and commit again.

If for some reason you want to skip pre-commit checks, you can use the `--no-verify` flag when committing (perhaps to correct formatting in a separate commit for clarity).

## Manually format
At the root of the project:
```julia
]
activate . # make sure you are in the CoupleNODE environement
# exit Pkg: hit backspace
using JuliaFormatter
format_file("path/to/file_to_format.jl")
```

## Manually create notebooks via `Literate.jl`
```julia
using Literate
Literate.notebook("path_to_file.jl"; execute = autorun_notebooks)
```
