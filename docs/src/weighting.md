# Weighting

`Weighting(; observations=:raw, rho_eps=nothing, target=:estimation)` configures
observation weighting and the default decomposition target.

Observation weighting controls how repeated firm-worker observations enter the
measurement model:

- `:raw` uses every observation row.
- `:edge` collapses repeated firm-worker pairs to edge means.
- `:effective` applies compound-symmetric residual effective weights using a
  fixed `rho_eps` or `rho_eps=:estimate`.

Variance decompositions can target `:estimation`, `:personyear`, or `:edge`.

## Grouped AR(1) observations

Under `Weighting(observations=:raw)`, `match_id` groups member rows into one
outcome and one residual per interval. Distinct-manager effects are averaged;
duplicating a member does not increase its weight. Controls retain the existing
first-member-row contract: use controls that agree within each match if row-order
invariance is required. No controls are silently dropped or regularized.

`error_eta=e` or `error_eta=:estimate` and `edge_index` define AR(1) over
successive **observed match ranks within each firm**. Each observed match must
have one firm and one rank; ranks must form `1:m` for that firm's observed
matches. This is not calendar-year decay or inverse-duration weighting.
Graph-only rows with nonfinite outcomes retain their prior adjacency links but
contribute neither observed matches nor ranks. Singleton observed firms have
residual correlation matrix `[1]` and contribute zero to its log determinant.

The likelihood dimension is the number `K` of observed groups, not the number
of manager-member rows. Its residual determinant is
`logdet(R) = (K - number_of_observed_firm_blocks) * log(1 - eta^2)`.
Both ExactCholesky and HutchSLQ support this specification with optional `X`.
AR(1) decomposition targets are currently unsupported and explicitly rejected.
