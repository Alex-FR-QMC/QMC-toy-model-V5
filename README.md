MCQ Experimental Engine

Experimental research repository for the MCQ (Meta-Cognitive Quantization / Modèle de Cognition Quantique) program.

This repository contains numerical prototypes, empirical protocols, and validation experiments designed to explore the operational consequences of the MCQ framework.

The objective is not to prove MCQ, but to build a progressive realizability map identifying which properties can or cannot emerge under increasingly expressive computational architectures.

---

Current Status

The first experimental program ("6d scalar branch") has been completed.

The 6d engine explored:

- scalar geometry "h(θ)"
- conformal-conservative diffusion
- single-instance dynamics ("N = 1")
- explicit Euler integration
- structured perturbation protocols
- basin analysis
- separability audits
- observability audits

The resulting experimental map is documented in the companion reports.

Main conclusion:

«The scalar architecture reached a stable descriptive ceiling.
No additional MCQ-positive properties appear reachable within the current assumptions.»

This conclusion motivates the transition toward the V5 architecture.

---

Repository Structure

/
├── engine/
│   ├── core numerical engine
│   ├── geometry operators
│   └── experimental utilities
│
├── protocols/
│   ├── perturbation protocols
│   ├── validation suites
│   └── basin analysis
│
├── reports/
│   ├── validation reports
│   ├── empirical audits
│   └── realizability maps
│
├── docs/
│   ├── MCQ notes
│   ├── architecture specifications
│   └── transition documents
│
└── archive/
    └── historical experiments

---

Methodological Principles

The project follows several strict principles.

Realizability before interpretation

Empirical results are evaluated before theoretical interpretation.

The primary question is:

«"What can this architecture actually produce?"»

not

«"How can this architecture confirm MCQ?"»

---

Explicit limitations

Every experimental branch is evaluated against its own limitations.

For the completed 6d branch:

- coarse spatial grid (5×5×5)
- single-instance architecture
- scalar metric only
- conformal approximation
- no explicit modular coupling operator

Negative results are considered informative.

---

Separation of debts

Architectural debts are isolated whenever possible.

Examples:

- coupling debt ("𝒞^{mod}")
- geometric debt ("H(θ)" tensor metric)
- multi-instance debt ("N > 1")

Future architectures are designed so these dimensions can be activated independently.

---

Transition Toward V5

The V5 program is intended to explore the next unresolved architectural questions.

Candidate directions include:

V5a

- multi-module dynamics
- explicit coupling operator "𝒞^{mod}"
- scalar geometry retained

Goal:

Isolate the effects of modular coupling.

V5b

- tensorial metric "H(θ)"
- geometric anisotropy
- transport effects
- Chapter 4 oriented investigations

Goal:

Explore non-scalar geometry.

No decision between V5a and V5b is assumed by this repository.

---

Scientific Position

This repository does not claim:

- proof of MCQ
- proof of consciousness
- proof of AGI
- proof of any ontological interpretation

It provides:

- computational experiments
- reproducible protocols
- empirical constraints
- realizability maps

Theoretical conclusions remain conditional on the architecture being tested.
