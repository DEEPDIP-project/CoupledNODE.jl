export helloworld

"""
    helloworld(text="World")

Return a greeting message.

Parameters:
    - `text::String`: The text to be greeted. Default is "World".

Returns:
    - `String`: A greeting message containing "Hello" followed by the input text and an exclamation mark.

Examples:
```julia
julia> helloworld()
"Hello World!"

julia> helloworld("Alice")
"Hello Alice!"

julia> helloworld("Bob")
"Hello Bob!"
"""
function helloworld(text::String = "World")
    return "Hello " * text * "!"
end
