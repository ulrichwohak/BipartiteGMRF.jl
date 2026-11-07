# Non-Backtracking Diagnostics

`nb_spectrum(problem)` computes the non-backtracking radius and localization
diagnostics of the binary prior graph. It can also operate directly on the
rectangular sparse matrix `problem.A_prior`.

```julia
spectrum = nb_spectrum(problem; seed=12345)
spectrum.lambda_nb
spectrum.in_two_core
spectrum.node_scores
```

The implementation peels the graph to its 2-core and solves each connected core
component separately through the sparse Ihara-Bass companion matrix. It does
not construct the directed-edge Hashimoto matrix and does not depend on a graph
package. Forests have `lambda_nb == 0`; a simple cycle has `lambda_nb == 1`.

Node-level vectors follow the package's latent ordering: firms first, then
workers. Localization scores are nonnegative and normalized within each core
component. Nodes in components without a cycle have `distance_to_core == -1`.

Large core components use Arnoldi iteration with a local seeded start vector.
Always check `spectrum.converged` before using the radius: a failed component
sets the global `lambda_nb` to `NaN` rather than returning a partial bound.

For `VarianceStablePrior(rho_limit=:auto)`, problem preparation uses this
diagnostic to resolve a guarded optimization limit. The default
`VarianceStablePrior()` remains the explicit numeric `0.99` behavior.

## Edge Pruning

`nb_prune_edges` can reduce the radius while preserving every graph component.
It protects a spanning forest and removes only cycle-closing edges from the
currently binding 2-core component.

```julia
prune = nb_prune_edges(df; target=3.0, seed=12345)
reduced = pruned_dataframe(df, prune)
```

`prune.keep` aligns with the original DataFrame rows, including repeated rows
for one edge. `prune.dropped_edges` reports unique removed edges and
`prune.audit` records the radius before and after each round. Check
`prune.target_met` before using the reduced graph.
