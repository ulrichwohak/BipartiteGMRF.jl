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
