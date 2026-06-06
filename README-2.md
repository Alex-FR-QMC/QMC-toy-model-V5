# MCQ / QMC Numerical Programme

This repository preserves the numerical and methodological artefacts of the **MCQ / QMC** project, especially the 6d scalar conformal-conservative programme and the transition material toward a possible V5 architecture.

The repository is not intended as a proof of MCQ. It is a reproducible workspace for testing what specific numerical instrumentations can or cannot make observable.

---

## 1. Scope

The current codebase documents the evolution of the **6d scalar programme**:

- scalar conformal metric `h(θ)`;
- finite-volume conservative diffusion;
- bounded perturbation protocols;
- basin diagnostics;
- dissociation diagnostics;
- limited reconstructibility audits;
- transition notes toward V5.

The 6d programme is now treated as **closed as an exploration of the instrumented scalar conformal-conservative engine**. It does not validate or invalidate MCQ, `𝒞^{mod}`, Ch3 self-opacity, or a future tensorial geometry `H(θ)`.

---

## 2. Methodological rule

The project follows a strict discipline:

> empirical result first, theoretical interpretation second.

All results should be classified as one of the following:

- **documented numerical fact**;
- **instrumentation-dependent observation**;
- **hypothesis for future architecture**;
- **unknown / unresolved tension**.

Avoid claims such as:

- “MCQ is proven”;
- “self-opacity is demonstrated”;
- “`𝒞^{mod}` is validated”;
- “the tensorial model is necessary.”

Acceptable formulations:

- “this operator produces / does not produce this observable under this instrumentation”;
- “this branch reveals a limit of the scalar conformal-conservative engine”;
- “this result motivates a future architectural test.”

---

## 3. Main empirical invariants inherited from 6d

The following invariants should be preserved as design constraints for any future V5 work.

### I₁ — Basins

6d produced robust basin-level distinctions under controlled initial families. Future models should keep basin diagnostics, trajectory distances, reactivation tests, and final-state comparisons.

### I₂ — Functional dissociation

6d revealed non-generic dissociation between `h-active` and `grad-active` regions, especially in basin-specific contexts. Future models should preserve diagnostics such as:

- `frac_h_active`;
- `frac_grad_active`;
- `intersection(h-active, grad-active)`;
- Jaccard overlap;
- gradient degeneracy status.

### I₃ — Structured dispersion

P4-VAR showed that the P4 cloud was not amorphous. Its dispersion was stratified by regimes such as intra-trajectory pairs, inter-basin pairs, B3 degeneracy, and B2/dissociation classes. Future models should keep variance audits rather than treating noisy clouds as automatically meaningless.

---

## 4. 6d engine summary

The 6d engine is a deliberately minimal numerical model:

```text
∂t ψ = diffusion conformal-conservative + optional drift / perturbations
∂t h = sedimentation + erosion
```

Core properties:

- grid: initially `5×5×5`;
- boundary condition: Neumann zero-flux;
- fluxes: conservative finite-volume fluxes;
- metric: scalar conformal `h(θ)`;
- face coefficients: harmonic mean of neighboring `h` values;
- time stepping: explicit Euler in the reference implementation;
- initial focus: clarity, debuggability, and reproducibility.

Known limits:

- no native multi-module coupling `𝒞^{mod}`;
- no multi-instance coupling `𝒞^N`;
- no tensorial metric `H(θ)`;
- no strict Laplace-Beltrami implementation;
- no transport parallelism;
- no native non-gradient circulation operator.

---

## 5. Transition status

The transition document `6d_to_V5_transition.md` frames the closure of 6d as a **maturity point**, not as an exhaustive negative result.

The remaining debts are architectural:

- native modular coupling `𝒞^{mod}`;
- multi-instance coupling `𝒞^N`;
- tensorial geometry `H(θ)`;
- operators beyond the scalar conformal-conservative approximation.

These are not “forgotten tests” inside 6d. The 6d engine does not have the degrees of freedom required to test them properly.

---

## 6. V5 direction

The current V5 planning material recommends starting with architecture, not phenomenology.

Recommended sequence:

```text
V5-0  architecture of objects and operator contracts
V5-1  compatibility mode reproducing 6d
V5-2  scalar multi-module model with native 𝒞^{mod}
V5-3  circulation / non-gradient audits
V5-4  tensor metric prototype or advanced support geometry
```

The V5 architecture should separate:

- `Grid`;
- `State`;
- `Metric`;
- `Module`;
- `Instance`;
- `GeometryOperator`;
- `CouplingScheduler`;
- `CouplingContext`;
- `CouplingOperator`;
- `Solver`;
- `Diagnostics`;
- `ExperimentProtocol`.

Key rule:

> `CouplingOperator` must not receive the full `Instance` object.

It should receive explicitly bounded `CouplingContext` objects constructed by a scheduler. This prevents hidden global averaging, consensus bias, and centralization leakage.

---

## 7. Repository organisation

A suggested organisation is:

```text
.
├── README.md
├── docs/
│   ├── 6d_to_V5_transition.md
│   ├── V5_0_architecture_spec.md
│   └── reports/
├── src/
│   └── mcq_v4/
│       └── factorial_6d/
├── tests/
│   ├── phase6d/
│   └── v5_architecture/
├── experiments/
│   ├── scripts/
│   └── outputs/
└── results/
    ├── json/
    └── figures/
```

This layout is only a proposal. Keep historical scripts if they are needed for reproducibility.

---

## 8. Reproducibility principles

Every experiment should provide:

- script name;
- input parameters;
- random seed if any;
- grid size;
- solver mode;
- diagnostic list;
- JSON output;
- human verdict separated from raw measurements.

Preferred pattern:

```text
specification document → script → JSON output → human audit → cadrage update
```

---

## 9. Guardrails

Do not silently change:

- grid resolution;
- boundary conditions;
- solver type;
- perturbation protocol;
- thresholds;
- active diagnostics;
- interpretation labels.

When changing any of these, create a new specification note or clearly document the change in the experiment output.

---

## 10. License / status

This repository is a research workspace. Choose an explicit license before public release.

Suggested options:

- MIT License for code if broad reuse is desired;
- CC BY-NC-SA for theoretical documents if non-commercial sharing is preferred;
- private repository until the theoretical framing is ready.

---

## 11. Short project statement

MCQ / QMC numerical work is not a search for confirmation. It is a sequence of controlled instruments designed to reveal which structures become observable under which constraints, and which debts require a different architecture.

The 6d branch mapped the scalar conformal-conservative regime. V5, if opened, must begin by preserving that map before extending it.
