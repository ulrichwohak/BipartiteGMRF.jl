---
name: testing
description: Use when writing, reviewing, or modifying tests in this repository, especially Julia Test-based testsets, TDD workflow, test organization, and behavior coverage.
---

# Testing Conventions

## TDD Approach

We follow Test-Driven Development (TDD):

1. **Write tests first** - Define expected behavior before implementation
2. **Run tests** - Verify they fail (red)
3. **Implement** - Write minimal code to pass tests (green)
4. **Refactor** - Clean up while keeping tests passing

## Test Structure

Tests live in `test/` directory, mirroring the `src/` structure:

```
test/
└── sample.jl          # Tests for src/create/sample.jl
```

## Writing Tests

Use `Test` standard library with `@testset` macros:

```julia
using Test
using Graphs

include("../src/create/sample.jl")

@testset "sample(::AbstractGraph, ::Sampler)" begin
    g = path_graph(5)

    @testset "Sampler(:full) returns full graph" begin
        sampler = Sampler(:full, 5, 4)
        result = sample(g, sampler)
        @test nv(result) == 5
        @test ne(result) == 4
    end
end
```

## Test Organization

- One top-level `@testset` per function/method being tested
- Nested `@testset` for specific behaviors/scenarios
- Descriptive test names that explain the expected behavior

## Running Tests

```bash
julia --project test/sample.jl
```

Or from Julia REPL:
```julia
using Pkg
Pkg.test()
```

## Do Not Implement to Pass Tests

When writing tests, **do not** modify implementation to make tests pass. The test file should be written independently, and the implementation should be updated separately to satisfy the tests.

## What to Test

- Happy paths (normal usage)
- Edge cases (empty inputs, boundary values)
- Error conditions (expected failures)
- Type stability (return types match annotations)
