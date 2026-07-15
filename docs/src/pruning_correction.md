# Why Pruned Rho Is Not Mechanically Corrected

Edge pruning changes the variance-stable kernel through both its adjacency and
degree terms. That perturbation is useful for diagnostics, but it does not by
itself identify a correction that turns an estimate of ``\rho`` on a pruned
graph into the estimate that would have been obtained on the full graph.

## Complete Kernel Perturbation

Write the full-graph kernel as

```math
B(\rho)=I-\rho A+\rho^2(D-I).
```

For a set of removed undirected edges ``R``, define

```math
E_R=\sum_{(u,v)\in R}(e_ue_v^\top+e_ve_u^\top),\qquad
\Delta_R=\sum_{(u,v)\in R}(e_ue_u^\top+e_ve_v^\top).
```

The pruned graph has ``A^-=A-E_R`` and ``D^-=D-\Delta_R``. Its exact kernel
difference at fixed ``\rho`` is therefore

```math
\delta B_{\mathrm{remove}}
  =B^-(\rho)-B(\rho)
  =\rho E_R-\rho^2\Delta_R.
```

Equivalently, when adding the dropped edges back to the pruned graph,

```math
\delta B_{\mathrm{add}}
  =B(\rho)-B^-(\rho)
  =-\rho E_R+\rho^2\Delta_R.
```

Keeping only the off-diagonal adjacency term is incomplete. Every removed edge
also lowers two node degrees, producing the diagonal term with the opposite
sign.

If ``K=B^{-1}``, the first-order inverse perturbation for either convention is

```math
\delta K=-K(\delta B)K+O(\lVert\delta B\rVert^2).
```

For the implied correlation

```math
c_{uv}=\frac{K_{uv}}{\sqrt{K_{uu}K_{vv}}},
```

the corresponding first-order change includes numerator and marginal-variance
terms:

```math
\delta c_{uv}
=\frac{\delta K_{uv}}{\sqrt{K_{uu}K_{vv}}}
-\frac{c_{uv}}{2}
 \left(\frac{\delta K_{uu}}{K_{uu}}+
       \frac{\delta K_{vv}}{K_{vv}}\right).
```

These formulas describe a fixed-parameter graph perturbation. They are not an
estimator correction, and the linear approximation can be poor when many
localized 2-core edges are removed.

## What A Rho Correction Would Require

Let ``\theta=(\rho,\psi)`` collect ``\rho`` and all nuisance parameters, and
let ``s(\theta;G)`` be the joint likelihood score on graph and observation
design ``G``. Linearizing the first-order conditions around a fitted model
would require

```math
H\,\delta\theta + D_Gs[\delta G]=0,\qquad
\delta\theta=-H^{-1}D_Gs[\delta G],
```

where ``H=\partial s/\partial\theta^\top`` is the full joint score Jacobian or
likelihood Hessian. Extracting only the ``\rho`` coordinate still requires the
cross-derivatives with the variance components and any other nuisance
parameters. A scalar adjustment would need an additional identifying
assumption, such as valid nuisance-score orthogonality or an explicitly derived
profile-score approximation. Neither is currently established.

There is a second distinction in estimation workflows: dropping graph edges
usually drops the corresponding DataFrame rows. This changes the observation
incidence matrix, response projection, sample size, and likelihood terms in
addition to changing ``A`` and ``D``. A perturbation of ``B(\rho)`` alone
cannot represent those changes.

## Recommended Comparison Target

Use [`implied_corr_functional`](@ref) as the reported correlation target rather
than applying an unverified adjustment to ``\rho``. It averages model-implied
matched-edge correlations and is marked `:exact` or `:stratified` according to
how outside-core edges were evaluated. On forests it reduces exactly to
``\rho``; on cyclic graphs it retains the additional non-backtracking paths and
reports core, adjacent, and tree strata through [`implied_correlations`](@ref).

This is the pruning-stable comparison supported by the current downstream
checks. It does not claim that pruning leaves the functional invariant; it
provides a common, topology-aware estimand whose movement can be measured
directly. A rho correction should remain deferred until the joint
score/Hessian calculation and its identifying assumption are supplied.
