# Non-Backtracking Diagnostics

`nb_spectrum(A_prior)` computes the non-backtracking radius and localization
diagnostics of the binary prior graph from a rectangular sparse adjacency
matrix (for a computed sufficient-statistics object, pass `ss.A_prior`).

```julia
spectrum = nb_spectrum(ss.A_prior; seed=12345)
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

For `BipartiteVarianceStableModel(A; rho_limit=:auto)` (or
`fit_mle(BipartiteVarianceStableModel, f, w, y; rho_limit=:auto)`), model
construction uses this diagnostic to resolve a guarded optimization limit
and stores the spectrum on the model. The default numeric `rho_limit=0.99`
remains the explicit behavior. `feasibility(model)` audits an explicit limit
against the ceiling after the fact.
