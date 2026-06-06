# V5-0 — Architecture Specification

**Status:** dry engineering specification.  
**Scope:** architecture, responsibilities, interfaces, tests, and epistemic guardrails.  
**Out of scope:** phenomenological claims, MCQ validation, Ch4 validation, full tensorial geometry, final discretization of \(\mathcal{C}^{mod}\), Crank–Nicolson/ADI implementation.

**Revision note:** this version integrates four V5-0 hardening amendments: differential routing of `kind="flux"`, macro-step-only history access, scheduler-owned topological tolerance near overlap limits, and explicit acknowledgement of scheduler performance debt for large modular graphs.

---

## 0. Purpose

V5-0 defines the minimal software and epistemic architecture required before any new dynamics are coded.

The objective is not to produce a new numerical result. The objective is to build a code structure in which each architectural debt identified after 6d can later be activated, deactivated, and audited without collapsing the effects into one another.

V5-0 therefore implements **contracts**, not yet a full physical model.

Central rule:

> V5 must be designed so that each architectural debt can be activated or deactivated separately, even if their theoretical independence is not guaranteed.

The debts to preserve as separable experimental axes are:

- modular coupling \(\mathcal{C}^{mod}\),
- tensorial geometry \(H(\theta)\),
- non-gradient circulation,
- multi-instance coupling \(\mathcal{C}^{N}\),
- resolution/support beyond the 5×5×5 scalar 6d grid.

---

## 1. Heritage from 6d: invariants to preserve

V5 is not a restart from zero. It must inherit the empirical and methodological constraints produced by 6d.

### I₁ — Basins

6d documented empirical basin structure. V5 must preserve basin diagnostics:

- final-state distances,
- trajectory distances,
- AUC of morphological divergence,
- family-dependent convergence,
- distinction between empirical basin and structural basin.

**Design consequence:** every V5 experiment must be able to compare families of initial states and determine whether basins persist, dissolve, or transform when \(\mathcal{C}^{mod}\) or \(H(\theta)\) is activated.

---

### I₂ — Functional dissociation

6d documented that the dissociation between active geometry and active gradients is not generic. It is basin-dependent.

V5 must preserve diagnostics such as:

- `frac_h_active`,
- `frac_grad_active`,
- `frac_intersection_h_grad`,
- `jaccard_h_grad`,
- `grad_status`,
- B3-like degeneracy detection.

**Design consequence:** V5 must never reduce dynamics to global active/inactive labels. It must measure where presence, transport, metric accessibility, and gradients coincide or separate.

---

### I₃ — Structured dispersion

P4-VAR showed that a dispersed observable/state cloud can still be stratified.

V5 must preserve:

- residual analysis,
- intra-trajectory vs inter-basin separation,
- pair-class variance audits,
- class-wise regressions,
- second-order structure in apparently noisy clouds.

**Design consequence:** absence of a first-order law is not automatically failure. It must be audited for residual stratification.

---

## 2. Epistemic guardrails

V5 must not claim:

- MCQ validation,
- Ch4 validation,
- self-opacity proof,
- \(\mathcal{C}^{mod}\) validation,
- tensorial necessity,
- AGI relevance.

Allowed language:

- “documents under this instrumentation,”
- “is compatible with,”
- “fails under this protocol,”
- “produces / does not produce an observable structure,”
- “requires further architecture to test.”

Forbidden shortcuts:

- turning numerical instability into theoretical emergence,
- turning no-signal into falsification,
- turning circulation into MCQ proof,
- turning a V5a/V5b option into a necessity claim.

---

## 3. Top-level architecture

V5-0 defines the following object families:

```text
Grid
State
Metric
Module
Instance
GeometryOperator
CouplingScheduler
CouplingContext
CouplingOperator
CouplingAggregator
Solver
Diagnostics
ExperimentProtocol
```

Each object has a restricted responsibility. No object should silently assume the role of another.

---

## 4. Responsibility graph

```text
ExperimentProtocol
 ├─ configures Grid, initial states, metrics, operators, solver, diagnostics
 ├─ launches experiments
 └─ exports results

Instance
 └─ contains Modules

Module
 ├─ owns State
 ├─ owns Metric
 └─ exposes controlled views

CouplingScheduler
 ├─ reads Instance
 ├─ builds authorized CouplingContext objects
 ├─ chooses admissible source-target pairs
 └─ sends contexts to CouplingOperator

CouplingOperator
 ├─ never sees full Instance
 ├─ receives only local CouplingContext
 ├─ computes pairwise C_{target←source}^{mod}
 └─ returns CouplingContribution

CouplingAggregator
 ├─ aggregates pairwise contributions per target module
 ├─ applies conservation/bounding policy
 └─ returns module-level coupling terms

GeometryOperator
 ├─ computes gradients, fluxes, divergences
 ├─ uses Metric coefficients
 └─ does not compute modular coupling

Solver
 ├─ receives geometry terms + coupling terms
 ├─ integrates time
 └─ does not define physics

Diagnostics
 ├─ observes states and fluxes
 └─ never modifies dynamics
```

---

## 5. Grid

### Purpose

Define the discrete support on which fields, fluxes, metrics, and operators live.

### Requirements

`Grid` must provide:

- shape,
- dimension,
- spacing,
- coordinates,
- cell centers,
- faces/interfaces,
- boundary conditions,
- neighbor access,
- optional staggered layout.

### Minimal interface

```python
class Grid:
    shape: tuple[int, ...]
    dx: float | tuple[float, ...]
    dim: int

    def cell_centers(self): ...
    def faces(self, axis: int): ...
    def neighbors(self, index): ...
    def boundary_policy(self): ...
```

### V5-0 decision

The grid must not hardcode 5×5×5 or 7×7×7. It must support both.

Initial recommended test grid:

```text
shape = (5, 5, 5) for 6d compatibility
shape = (7, 7, 7) for V5 exploratory support tests
```

Staggered grid support is optional in V5-0 but should not be made impossible by the design.

---

## 6. State

### Purpose

Hold dynamical fields and metadata without becoming opaque.

### Requirements

A `State` must be observable and explicit. It must not hide fields behind excessive magic methods.

### Minimal interface

```python
class State:
    fields: dict[str, np.ndarray]
    metadata: dict
    history_hooks: object | None

    def get_field(self, name: str) -> np.ndarray: ...
    def with_field(self, name: str, value: np.ndarray) -> "State": ...
    def copy(self) -> "State": ...
    def flatten_for_solver(self) -> np.ndarray: ...
    def norm(self, fields: list[str] | None = None) -> float: ...
```

Allowed controlled algebra:

```python
state_scaled = state.scale(a)
state_sum = state.add(other)
state_diff = state.subtract(other)
```

Avoid unrestricted magic methods in V5-0 unless they remain transparent and tested.

### Rationale

6d diagnostics depended on opening the state directly:

- \(\psi\),
- \(h\) or \(H\),
- gradients,
- underflow,
- roughness,
- h-active/grad-active intersections.

V5 must preserve this inspectability.

---

## 7. Views: StateView, MetricView, HistoryView, GridView

### Purpose

Provide local, authorized access to data without giving operators the full global state.

### Rule

Views must be **zero-copy references** or lightweight indexed views. They must not allocate heavy objects inside the time loop.

### Minimal interfaces

```python
class StateView:
    def field(self, name: str) -> np.ndarray: ...
    def local_patch(self, radius: int): ...
    def diagnostics_allowed(self) -> dict: ...
```

```python
class MetricView:
    def local_coefficients(self): ...
    def distance(self, a, b, mode: str = "metric_l2"): ...
    def diagnostics_allowed(self) -> dict: ...
```

```python
class HistoryView:
    def previous(self, field: str): ...
    def window(self, field: str, length: int): ...
```

### 7.1 History closure condition

`HistoryView` exposes only states corresponding to **validated physical macro-steps**.

It must not expose:

- internal substeps of an adaptive solver,
- predictor/corrector intermediate states,
- provisional Crank–Nicolson or ADI states,
- rollback states,
- future states.

If a future solver uses substeps or adaptive timesteps, those states remain solver-internal until the macro-step is committed. Only committed states may enter `HistoryView`.

### Anti-leak requirement

A local operator must not access:

- the full `Instance`,
- experiment labels,
- all modules unless explicitly authorized,
- future states,
- full history by default,
- global averages unless explicitly included in the context.

---

## 8. Metric

### Purpose

Represent geometry. It does not compute dynamics by itself.

### Separation rule

`Metric` provides coefficients and distances.  
`GeometryOperator` computes gradients, fluxes, and divergences.

### Minimal interface

```python
class Metric:
    def face_coefficients(self, grid: Grid): ...
    def inverse(self): ...
    def determinant(self): ...
    def volume_element(self): ...
    def distance(self, state_a: State, state_b: State, mode: str = "metric_l2"): ...
    def update(self, psi: np.ndarray, dt: float): ...
    def diagnostics(self) -> dict: ...
    def view(self, selector=None) -> MetricView: ...
```

### Implementations

```python
class ScalarConformalMetric(Metric):
    # h(theta)
    ...
```

```python
class TensorMetric(Metric):
    # H(theta), SPD tensor field
    ...
```

### Distance modes

V5-0 must not require a true geodesic solver.

Required:

```text
mode="metric_l2"
```

Optional later:

```text
mode="path_approx"
mode="geodesic_approx"
```

### TensorMetric requirements, when activated

- SPD at every cell or face,
- eigenvalues bounded,
- stable inverse,
- determinant non-degenerate,
- underflow/overflow diagnostics,
- no silent clipping without diagnostic flag.

---

## 9. GeometryOperator

### Purpose

Apply differential operators using a metric and a grid.

### Minimal interface

```python
class GeometryOperator:
    def gradient(self, field: np.ndarray, metric: Metric, grid: Grid): ...
    def flux(self, field: np.ndarray, metric: Metric, grid: Grid): ...
    def divergence(self, flux, grid: Grid): ...
    def apply(self, state: State, metric: Metric, grid: Grid) -> dict[str, np.ndarray]: ...
```

### Requirements

- conservation of mass for conservative fluxes,
- clear boundary conditions,
- explicit separation of diffusion and drift,
- ability to reproduce 6d scalar conformal-conservative diffusion,
- no hidden modular coupling.

### 6d compatibility target

In mode:

```text
N = 1
M = 1
ScalarConformalMetric
CouplingOperator = Off
ExplicitReferenceSolver
```

`GeometryOperator` must reproduce the 6d scalar conformal-conservative flux logic.

---

## 10. Module and Instance

### Module

```python
class Module:
    id: ModuleId
    state: State
    metric: Metric
    metadata: dict

    def state_view(self, selector=None) -> StateView: ...
    def metric_view(self, selector=None) -> MetricView: ...
    def diagnostics(self) -> dict: ...
```

### Instance

```python
class Instance:
    id: InstanceId
    modules: dict[ModuleId, Module]
    metadata: dict

    def module_ids(self): ...
    def get_module(self, module_id: ModuleId) -> Module: ...
```

### Rule

`Instance` contains modules but does not compute coupling.

---

## 11. Coupling architecture

### 11.1 Anti-centralization principle

`CouplingOperator` must never receive the full `Instance`.

The allowed chain is:

```text
Instance → CouplingScheduler → CouplingContext(target←source) → CouplingOperator
```

### Rationale

This prevents clandestine centralization:

- no hidden global mean,
- no consensus operator,
- no global attractor,
- no experiment-label leak,
- no access to unauthorized histories.

---

### 11.2 CouplingContext

```python
@dataclass(slots=True)
class CouplingContext:
    target_id: ModuleId
    source_id: ModuleId
    overlap_R: float | np.ndarray
    target_state: StateView
    source_state: StateView
    target_metric: MetricView
    source_metric: MetricView
    grid: GridView
    time: float
    allowed_history: HistoryView | None = None
```

### Critical design choice: overlap type

`overlap_R` must allow both:

```text
float       # global overlap, V5a minimal
np.ndarray  # spatialized overlap field, V5b or advanced V5a
```

The interface must not require rewriting the operator when moving from global to local overlap.

### Epistemic caveat

A scalar global \(R_{ij}\) reintroduces a macro-property into local coupling. This is allowed for V5a minimal but must be flagged. A spatialized \(R_{ij}(x)\) is more local but more complex.

---

### 11.3 CouplingOperator

```python
class CouplingOperator:
    def pairwise_coupling(self, ctx: CouplingContext) -> "CouplingContribution": ...
```

It computes only:

```text
C_{target←source}^{mod}
```

It does not aggregate over all sources.

### Required abstract invariants

For a pair \(i ← j\):

- \(R_{ij} \to 0 \Rightarrow C_{i←j}^{mod} = 0\),
- \(R_{ij} \to 1 \Rightarrow C_{i←j}^{mod} = 0\),
- intermediate \(R_{ij}\) may produce non-zero coupling,
- novelty of source is evaluated in source metric,
- self-form of target is evaluated in target metric,
- \(C_{i←j}^{mod}\) may differ from \(C_{j←i}^{mod}\),
- coupling is not assumed to be a potential drift,
- coupling must not force consensus/fusion.

---

### 11.4 CouplingContribution typing

A coupling contribution must declare its mathematical type.

```python
@dataclass(slots=True)
class CouplingContribution:
    target_id: ModuleId
    source_id: ModuleId
    kind: Literal["source", "flux", "mixed"]
    value: np.ndarray | tuple[np.ndarray, ...]
    conservative: bool
    mass_delta: float
    diagnostics: dict
```

### Reason

\(\mathcal{C}^{mod}\) may be implemented as:

1. a direct source/sink term on \(\partial_t \psi_i\),
2. a conservative flux field,
3. a mixed contribution.

These have different conservation consequences.

---

## 12. CouplingScheduler

### Purpose

Orchestrate coupling without computing the pairwise physics.

```python
class CouplingScheduler:
    def build_contexts(self, instance: Instance) -> list[CouplingContext]: ...
    def admissible_pairs(self, instance: Instance) -> list[tuple[ModuleId, ModuleId]]: ...
```

Responsibilities:

- choose admissible pairs,
- compute or retrieve \(R_{ij}\),
- construct authorized views,
- attach allowed history if protocol permits it,
- ensure no unauthorized global information reaches `CouplingOperator`.

### 12.1 Topological tolerance

`CouplingScheduler` is the guardian of pair admissibility. It applies numerical cutoffs before any pair reaches `CouplingOperator`.

Required policy fields:

```text
epsilon_R_min
epsilon_R_max
```

Interpretation:

- if \(R_{ij} < \epsilon_{R,min}\), the pair is omitted;
- if \(1 - R_{ij} < \epsilon_{R,max}\), the pair is omitted or explicitly passed as a zero-coupling context for testing only;
- the chosen policy must be exported in protocol metadata.

Rationale: `CouplingOperator` should not decide whether a link exists. It should only compute coupling for admissible links. This prevents each coupling implementation from inventing its own numerical singularity handling near \(R 	o 0\) or \(R 	o 1\).

For spatialized overlap fields \(R_{ij}(x)\), the same rule applies elementwise or patchwise according to a declared policy.

---

## 13. CouplingAggregator

### Purpose

Aggregate pairwise contributions per target module while making conservation policy explicit.

```python
class CouplingAggregator:
    def aggregate(
        self,
        contributions: list[CouplingContribution],
        policy: str,
    ) -> dict[ModuleId, CouplingContribution]: ...
```

### 13.1 Differential routing of coupling contributions

`Solver` must receive unified state-derivative terms. It must not compute spatial divergence itself.

Routing rule:

```text
CouplingOperator -> CouplingContribution
CouplingAggregator -> aggregate by target module
GeometryOperator.divergence(flux, grid) -> convert kind="flux" into scalar ∂tψ contribution
Solver -> integrate scalar/vector state derivatives only
```

If a contribution has `kind="flux"`, the aggregated value is a vector/face field. `CouplingAggregator` routes this field to `GeometryOperator.divergence(flux, grid)` using the declared boundary policy. The result is then packaged as a scalar source-like derivative for the target module.

If a contribution has `kind="source"`, no divergence is applied. Its mass delta must be reported.

If a contribution has `kind="mixed"`, its flux and source parts must be separated before integration.

This preserves the division of roles:

- `CouplingOperator` computes pairwise modular physics;
- `CouplingAggregator` aggregates and routes;
- `GeometryOperator` owns discrete differential operations;
- `Solver` integrates time and does not own stencil logic.

### Required policies

```text
policy="conservative"
policy="source_allowed"
policy="bounded_source"
policy="diagnostic_only"
```

### Conservation tension

V5-0 must not silently decide whether \(\mathcal{C}^{mod}\) conserves mass.

Two hypotheses remain active:

#### Hypothesis A — open modular dynamics

\(\mathcal{C}^{mod}\) may act as a source/sink of informational mass at module level.

#### Hypothesis B — hard conservation

The aggregated modular coupling must satisfy:

```text
Σ_i C_i^{mod} = 0
```

or be represented as a conservative flux.

### V5-0 decision

The type of contribution and policy must be explicit in the exported diagnostics.

No bounded or saturated coupling may be applied without reporting:

- pre-bound mass delta,
- post-bound mass delta,
- norm before bounding,
- norm after bounding,
- cells/modules affected.

---

## 14. Solver

### Purpose

Integrate time. It does not define physics.

### Initial implementation

```python
class ExplicitReferenceSolver:
    def step(self, instance, operators, dt): ...
```

### Later stubs

```python
class SemiImplicitSolver: ...
class ADISolver: ...
class CrankNicolsonSolver: ...
```

### Rule

No advanced solver before:

1. operators are isolated,
2. explicit reference mode exists,
3. conservation/positivity tests pass,
4. 6d compatibility mode is reproduced.

### Rationale

A semi-implicit solver can hide high-frequency bugs, tensor instabilities, and coupling pathologies. The explicit solver is a crash-test.

---

## 15. Diagnostics

### Inherited diagnostics

- mass conservation,
- positivity,
- underflow,
- h_min / H eigenvalue minima,
- roughness,
- basin distances,
- AUC trajectory distances,
- h-active / grad-active dissociation,
- P4VAR-style residual stratification,
- reactivation diagnostics,
- top ambiguous pairs,
- intra/inter class separation.

### New V5 diagnostics

- overlap \(R_{ij}\),
- spatialized overlap if available,
- novelty transmitted,
- fusion metrics,
- non-fusion persistence,
- pairwise coupling norms,
- aggregate coupling norms,
- mass delta by coupling kind,
- gradient flux vs coupling flux,
- curl / circulation of flux,
- boundary circulation controls,
- stencil artefact controls.

### Circulation guardrail

Allowed conclusion:

> activating \(\mathcal{C}^{mod}\) documents a non-gradient circulation component under this instrumentation.

Forbidden conclusion:

> circulation proves MCQ.

---

## 16. ExperimentProtocol

### Purpose

Make experiments first-class objects.

```python
class ExperimentProtocol:
    def build_grid(self) -> Grid: ...
    def build_initial_states(self) -> Instance: ...
    def configure_metrics(self) -> dict: ...
    def configure_operators(self) -> dict: ...
    def configure_solver(self) -> Solver: ...
    def configure_diagnostics(self) -> Diagnostics: ...
    def run(self): ...
    def export(self, path: str): ...
```

### Required protocol metadata

- protocol name,
- version,
- grid shape,
- metric type,
- number of instances,
- number of modules,
- active operators,
- coupling policy,
- conservation policy,
- history policy,
- seeds,
- checkpoints,
- guardrails.

---

## 17. V5-0 unit tests

### 17.1 Visibility tests

- `CouplingOperator` does not receive `Instance`.
- `CouplingOperator` does not receive `ExperimentProtocol`.
- `CouplingOperator` does not receive family labels.
- Changing forbidden global metadata does not change coupling.

### 17.2 Coupling abstract tests

- \(R=0\) ⇒ coupling norm = 0.
- \(R=1\) ⇒ coupling norm = 0.
- intermediate \(R\) + novelty ⇒ coupling norm > 0.
- source metric change modifies novelty.
- target metric change modifies self-form.
- \(i←j\) can differ from \(j←i\).
- no automatic fusion under repeated coupling-only steps.

### 17.3 Conservation and routing tests

For every `CouplingContribution`:

- report mass delta,
- report conservative flag,
- test conservative policy if selected,
- test source_allowed policy separately,
- ensure bounded_source reports pre/post norms,
- verify `kind="flux"` is routed through `GeometryOperator.divergence`,
- verify `Solver` receives only unified derivative terms and no raw flux field,
- verify mixed contributions are split before integration.

### 17.3bis History and tolerance tests

- `HistoryView` exposes only committed macro-step states.
- internal solver substeps do not enter `HistoryView`.
- `CouplingScheduler` omits pairs with \(R < \epsilon_{R,min}\).
- `CouplingScheduler` omits or zero-flags pairs with \(1-R < \epsilon_{R,max}\).
- changing the epsilon policy changes admissible pairs but not the local implementation of `CouplingOperator`.

### 17.4 Metric tests

Scalar metric:

- reproduce 6d harmonic-face behavior,
- positivity,
- no silent zeros,
- correct diagnostics.

Tensor metric stub:

- SPD checks,
- eigenvalue bounds,
- inverse stability,
- determinant positivity.

### 17.5 Geometry tests

- constant field has zero flux,
- Neumann boundary conserves mass,
- gradient-only flux has negligible curl,
- scalar compatibility with 6d diffusion.

### 17.6 Performance tests

- views are zero-copy or lightweight,
- no per-cell Python object allocation inside inner loops,
- no per-step heavy context reconstruction if contexts can be cached,
- allocation counts monitored in smoke tests.

---

## 18. Performance requirements

V5 must avoid object overhead in the inner time loop.

### Rules

- `StateView`, `MetricView`, `GridView` use references/slices/indices.
- contexts use `dataclass(slots=True)` or equivalent.
- admissible pair lists may be precomputed.
- static overlap topology may be cached.
- NumPy arrays remain the numerical substrate in V5-0.
- no Python object per cell.
- no dynamic allocation in the innermost flux computation.

### Performance caveat

Architecture must protect epistemic clarity without making the simulator unusable.

The pairwise scheduler has an expected cost of \(M(M-1)\) contexts per instance and per macro-step. For V5-2 with \(M=2\) or \(M=3\), this cost is accepted in favor of clarity. For larger modular networks, this becomes a known performance debt and may require compiled scheduling, cached context pools, or vectorized pair batches.

This performance debt must not be solved by giving `CouplingOperator` access to `Instance`. Performance optimization must preserve the anti-centralization boundary.

---

## 19. V5-0 implementation sequence

### V5-0a — UML and interfaces

Deliver:

- `grid.py`
- `state.py`
- `metric.py`
- `module.py`
- `instance.py`
- `operators.py`
- `coupling_context.py`
- `coupling_scheduler.py`
- `coupling_operator.py`
- `solver.py`
- `diagnostics.py`
- `experiment_protocol.py`
- `tests/`

No phenomenology.

---

### V5-0b — Scalar compatibility skeleton

Implement:

- `ScalarConformalMetric`,
- `GeometryOperator` reproducing 6d diffusion,
- `ExplicitReferenceSolver`,
- M=1, N=1 compatibility mode.

Goal:

```text
V5 reproduces 6d baseline when all new architecture is turned off.
```

---

### V5-0c — Coupling dry tests

Implement synthetic `CouplingContext` tests without full simulation.

Goal:

```text
Coupling invariants pass before any coupling is integrated in time.
```

---

### V5-1 — 6d reproduction

Full reproduction of selected 6d references.

---

### V5-2 — First \(\mathcal{C}^{mod}\) prototype

N=1, M=2 or M=3, scalar metric, pairwise modular coupling active.

---

### V5-3 — Circulation audit

Measure whether modular coupling produces non-gradient circulation under controlled conditions.

---

### V5-4 — TensorMetric prototype

Introduce tensorial metric only after scalar modular coupling and diagnostics are stable, unless a separate decision explicitly chooses V5b direct.

---

## 20. Final V5-0 decision

The next deliverable should be:

```text
V5_0_architecture_spec.md
```

It should not yet include:

- final \(\mathcal{C}^{mod}\) discretization,
- Crank–Nicolson or ADI,
- full tensorial Ch4,
- multi-instance dynamics,
- MCQ claims.

It must include:

- object responsibilities,
- anti-centralization coupling architecture,
- conservation/source typing,
- routing of flux contributions through `GeometryOperator.divergence`,
- overlap scalar/local indeterminacy,
- scheduler-owned topological tolerance near \(R 	o 0\) and \(R 	o 1\),
- macro-step-only history access,
- zero-copy view requirement,
- known scheduler performance debt for large \(M\),
- unit tests,
- guardrails,
- implementation sequence.

---

## 21. Closing formulation

V5-0 is not yet a model. It is the architecture that prevents the next model from becoming unreadable.

Its function is to preserve experimental separability without pretending theoretical separability is guaranteed.

The correct next step is therefore:

```text
write V5_0_architecture_spec.md
then implement V5-0a
then reproduce 6d before activating any new physics
```
