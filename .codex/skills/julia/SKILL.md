---
name: julia
description: Use when writing, reviewing, or modifying Julia code in this repository, especially scripts, package internals, CLI entrypoints, logging, file IO, path handling, and Julia style conventions.
---

# Julia Coding Conventions

## Project Setup

This project uses Julia with a standard project environment structure. Scripts assume the project environment is already activated when run:

```bash
julia --project src/create/script.jl arg1 arg2
```

Do **not** include `Pkg.activate(".")` inside scripts.

## Required Dependencies

Common packages used in this project:
- `Parquet2` - Reading/writing Parquet files
- `DataFrames` - Tabular data manipulation
- `Logging` - Structured logging (not println)
- `Kezdi.jl` - Data manipulation (when available)

## CLI Argument Handling

Scripts that accept command-line arguments should:

1. Include a docstring with usage information
2. Validate required arguments
3. Use proper exit codes
4. Use `@error` and `@info` for messaging

```julia
#!/usr/bin/env julia

"""
script.jl - Brief description

Usage:
    julia script.jl <arg1> <arg2>

Arguments:
    arg1    Description of arg1
    arg2    Description of arg2

Input:
    temp/input.parquet

Output:
    temp/output.parquet
"""

using Parquet2
using DataFrames
using Logging

function main()
    args = ARGS

    if length(args) < 2
        @error "Missing required arguments"
        @info "Usage: julia script.jl <arg1> <arg2>"
        exit(1)
    end

    arg1 = args[1]
    arg2 = args[2]

    # ... implementation
end

main()
```

## Logging

Always use the `Logging` module instead of `println`:

- `@error "message"` - For errors (goes to stderr)
- `@warn "message"` - For warnings
- `@info "message" key=value` - For informational messages with structured data
- `@debug "message"` - For debug output (only shown with appropriate log level)

Benefits:
- Structured output with key-value pairs
- Configurable log levels
- Proper separation of stdout/stderr
- Timestamp and source line info

## File I/O Patterns

### Reading Parquet
```julia
using Parquet2
using DataFrames

df = DataFrame(Parquet2.readfile("temp/input.parquet"))
```

### Writing Parquet
```julia
Parquet2.writefile("temp/output.parquet", df)
```

### Directory Creation
```julia
mkpath("temp/subdir")  # Creates parent directories as needed
```

### File Existence Check
```julia
isfile(path) || (@error "Input file not found" path; exit(1))
```

## Path Conventions

- All paths relative to project root
- Use `joinpath()` for cross-platform compatibility
- Input files: `temp/` or `input/`
- Output files: `temp/` or `output/`
- Follow directory structure: `temp/<stage>/<param>/file.parquet`

## Error Handling

- Validate inputs early
- Use descriptive error messages with `@error`
- Exit with non-zero code on failure
- Include relevant context as key-value pairs

```julia
if !isfile(input_path)
    @error "Input file not found" input_path expected_location="temp/"
    exit(1)
end
```

## Julianic Style Conventions

These are non-negotiable idioms that make code feel native to Julia.

### One-Line Functions with Return Type Annotations

For simple functions, use one-line syntax with explicit return types:

```julia
# Good
sample(g::AbstractGraph, sampler::Sampler)::AbstractGraph =
    NotImplementedError("sample") |> throw

# Bad - multi-line for simple functions
function sample(g::AbstractGraph, sampler::Sampler)
    throw(NotImplementedError("sample"))
end
```

### Short-Circuit Guard Clauses

Use short-circuit evaluation (`&&`/`||`) for guard clauses and early returns:

```julia
# Good - short-circuit guard clauses
sample(g::AbstractGraph, sampler::Sampler)::AbstractGraph = begin
    sampler.method == :full && return g
    sampler.method == :bfshop && return _sample_bfshop(g, sampler)
    NotImplementedError("unknown method") |> throw
end

bfs_dhop(g::SimpleGraph, start::Integer, d::Integer)::Set{Int} = begin
    d < 0 && ArgumentError("d must be non-negative") |> throw
    d == 0 && return Set{Int}([start])
    _bfs_dhop_impl(g, start, d)
end

# Bad - ternary chains
sample(g::AbstractGraph, sampler::Sampler)::AbstractGraph =
    sampler.method == :full ? g :
    sampler.method == :bfshop ? _sample_bfshop(g, sampler) :
    NotImplementedError("unknown method") |> throw
```

### Pipe Operator for Function Composition

Use `|>` instead of nesting function calls:

```julia
# Good
error("message") |> throw
result = data |> process |> analyze

# Bad - nested calls
throw(error("message"))
result = analyze(process(data))
```

### Multiple Dispatch, Not Type Names in Functions

Use the same function name with different argument types:

```julia
# Good - multiple dispatch
sample(lg::LabeledGraph, sampler::Sampler)::LabeledGraph = ...
sample(g::AbstractGraph, sampler::Sampler)::AbstractGraph = ...

# Bad - type names in function names
sample_labeledgraph(lg, sampler)
sample_abstractgraph(g, sampler)
```

### Avoid Private Helper Functions

Don't split into `validate_then_call` patterns. Use validation + logic in one function with early returns:

```julia
# Good - single function with validation
function bfs_dhop(g::SimpleGraph, start::Integer, d::Integer)::Set{Int}
    d < 0 && ArgumentError("d must be non-negative") |> throw
    # ... implementation
    visited
end

# Bad - private helper pattern
bfs_dhop(g::SimpleGraph, start::Integer, d::Integer)::Set{Int} = begin
    d < 0 && ArgumentError("d must be non-negative") |> throw
    _bfs_dhop_impl(g, start, d)
end
function _bfs_dhop_impl(g::SimpleGraph, start::Integer, d::Integer)::Set{Int}
    # ... implementation
end
```

### Type Constructors for Conversion

Use constructors to convert between types:

```julia
# Good - constructor pattern
DataFrame(lg::LabeledGraph)::DataFrame = ...
LabeledGraph(df::DataFrame, src::Symbol, tgt::Symbol)::LabeledGraph = ...

# Bad - explicit convert function names
dataframe_from_labeledgraph(lg)
edgelist_from_dataframe(df, src, tgt)
```

### Parametric Types for Generic Containers

Use parametric types when wrapping other types:

```julia
# Good - parametric type
struct LabeledGraph{G<:AbstractGraph}
    graph::G
    # ...
end

# Bad - concrete type
struct LabeledGraph
    graph::AbstractGraph
    # ...
end
```

### Omit Docstrings Until Needed

Don't write docstrings for obvious functions. We can always add them later.

```julia
# Good - just the code
sample(lg::LabeledGraph, sampler::Sampler)::LabeledGraph = ...

# Bad - docstring states the obvious
"""
    sample(lg::LabeledGraph, sampler::Sampler) -> LabeledGraph

Sample from a LabeledGraph using the specified sampler.
"""
sample(lg::LabeledGraph, sampler::Sampler)::LabeledGraph = ...
```

## Script Template

```julia
#!/usr/bin/env julia

"""
filename.jl - One-line description

Usage:
    julia filename.jl <required_arg> [optional_arg]

Arguments:
    required_arg    What it does
    optional_arg    What it does (default: value)

Input:
    temp/input.parquet

Output:
    temp/output.parquet
"""

using Parquet2
using DataFrames
using Logging

function main()
    args = ARGS

    # Validate arguments
    if length(args) < 1
        @error "Missing required argument"
        @info "Usage: julia filename.jl <required_arg>"
        exit(1)
    end

    param = args[1]

    # Define paths
    input_path = "temp/input.parquet"
    output_dir = joinpath("temp", "stage", param)
    output_path = joinpath(output_dir, "output.parquet")

    # Validate input
    if !isfile(input_path)
        @error "Input file not found" input_path
        exit(1)
    end

    # Create output directory
    mkpath(output_dir)

    # Process
    @info "Reading input" input_path
    df = DataFrame(Parquet2.readfile(input_path))

    # ... do work ...
    @info "Processing" param rows=nrow(df)

    # Write output
    @info "Writing output" output_path
    Parquet2.writefile(output_path, df)

    @info "Done" rows=nrow(df)
end

main()
```
