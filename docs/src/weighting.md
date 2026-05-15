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
