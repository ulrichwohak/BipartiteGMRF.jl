---
name: makefile
description: Use when creating, reviewing, or modifying Makefiles, especially pattern rules, automatic variables, second expansion, and path component extraction.
---

# Makefile Patterns and Conventions

## Pattern Rules

Use `%` for pattern matching in rules:

```makefile
temp/chunks/%/edgelist.parquet: temp/edgelist.parquet src/create/chunk.jl
	@$(JULIA) src/create/chunk.jl $*
```

The `$*` automatic variable captures the stem (the part matching `%`).

## Automatic Variables

- `$@` - The target filename
- `$<` - The first prerequisite
- `$*` - The stem (the part matching `%` in pattern rules)
- `$^` - All prerequisites (space-separated)
- `$?` - Prerequisites newer than target

## .SECONDEXPANSION

For complex prerequisite expansion (extracting parts of the stem), use `.SECONDEXPANSION:`:

```makefile
.SECONDEXPANSION:
temp/samples/%/edgelist.parquet: temp/chunks/$$(word 1,$$(subst /, ,$$*))/edgelist.parquet
	@$(JULIA) src/create/sample.jl $(word 1,$(subst /, ,$*)) $(word 2,$(subst /, ,$*))
```

Key points:
- Prerequisites use `$$` for deferred expansion (second expansion)
- Recipe uses `$` for normal expansion (at execution time)
- `$(word 1,$(subst /, ,$*))` extracts first path component
- `$(word 2,$(subst /, ,$*))` extracts second path component

## Suppressing Command Echo

Use `@` prefix to suppress printing the command:

```makefile
	@$(JULIA) script.jl arg  # Silent
	$(JULIA) script.jl arg   # Prints: julia --project=. script.jl arg
```

## Path Component Extraction

Extract parts of a path using `subst` and `word`:

```makefile
# For stem = random/train
$(word 1,$(subst /, ,$*))  # random
$(word 2,$(subst /, ,$*))  # train
$(subst /, ,$*)           # random train (space-separated list)
```

## Multi-Level Directory Targets

Pattern rules work well for directory structures:

```bash
# Build with:
make temp/chunks/random/edgelist.parquet
make temp/samples/random/train/edgelist.parquet
```

The stem `$*` captures:
- `random` for chunks
- `random/train` for samples
