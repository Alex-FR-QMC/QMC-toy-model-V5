# V5-0b — Closure report

**Date de clôture** : Tue, June 09, 2026
**Posture** : harnais strict minimal (Option 1 du protocole V5-0b)
**Préalable** : V5-0a fermée (14 étapes, 582 tests PASS, guards
appliqués)
**Référentiel** : `docs/V5_0b_6d_compatibility_protocol.md`

---

## §1. Verdicts stratifiés

| Bloc | Statut | Justification |
|---|---|---|
| `GEOMETRY_COMPAT` | **PASS** | 15/15 tests, `atol=1e-12`, cellule par cellule |
| `PROTOCOL_COMPAT` | **PASS** | 16/16 tests, dynamique préservée sous NullCoupling |
| `REFERENCE_6D_AVAILABLE` | mixte | 4 fichiers présents, schémas variants (voir §4) |
| `FULL_REPRODUCTION` | **OUT_OF_SCOPE** | h-dynamics absente de V5-0a |

Aucun verdict global unique. Chaque bloc est inscrit séparément, comme
exigé par le protocole §3.

---

## §2. État du source

**Aucun module `mcq_v5.*` n'a été modifié** pendant V5-0b.

Vérification par hash md5 :

```
src/mcq_v5/grid.py                d96632a47e035664221c77bfd1da55e9
src/mcq_v5/state.py               9fcd28a1f20ebc81e23f5253528c3362
src/mcq_v5/views.py               dc0da167fdf8327f13dc810fca680c9d
src/mcq_v5/metric.py              32baae1c3f044d72e68f5043582971a2
src/mcq_v5/module.py              a6f810365d71cec38b69ea1ae9865a34
src/mcq_v5/instance.py            9409fed851a4f7bed8ec59c949420528
src/mcq_v5/geometry.py            aa571f70633400c0a28af89a7d850cf1
src/mcq_v5/coupling_context.py    84f612ed12ce28df58768357cfed60b7
src/mcq_v5/coupling_scheduler.py  3bc30b53d9188be91e9b851fab844136
src/mcq_v5/coupling_operator.py   e64d759e4ef533fe8590f0db7546adeb
src/mcq_v5/coupling_aggregator.py c268813f3ee8ecf511649fa0ccfd974c
src/mcq_v5/solver.py              e8a41be976828b6cb2baf48be67ba633
src/mcq_v5/diagnostics.py         833ab8cc0c2e23394693629547f0587f
src/mcq_v5/experiment.py          7efdb8a536a0f1c044535aad97e32d59
```

Ces hashes sont **identiques** à ceux inscrits à la clôture de V5-0a.
La règle de non-modification est tenue.

---

## §3. Suite de tests V5-0b

| Fichier | Tests | Statut |
|---|---|---|
| `tests/test_v5_0b_geometry_compatibility.py` | 15 | PASS |
| `tests/test_v5_0b_protocol_compatibility.py` | 16 | PASS |
| `tests/test_v5_0b_reference_json_availability.py` | 10 | PASS |
| **Sous-total V5-0b** | **41** | **PASS** |
| Suite V5-0a (rappel) | 582 | PASS |
| **Total combiné** | **623** | **PASS** |

### §3.A — Geometry compatibility

`GeometryOperator` reproduit le cœur spatial 6d (harmonic-mean
face coefficients, flux `-h_face·grad(ψ)`, divergence `-div(J)`,
zero-flux Neumann implicite) à `atol=1e-12` cellule par cellule pour :
- champ constant 1D et 3D
- pic central [0,1,0] → [1,-2,1] et 3D centered pic [2,2,2]
- champ linéaire sous Neumann (RHS intérieur nul, contributions
  opposées aux bords, conservation globale)
- random déterministe 3D (seed=42), avec et sans anisotropic dx
- h variable déterministe 1D et 3D
- conservation `|sum(dψ)| < 1e-12` pour les 4 cas canoniques

Fonction de référence 6d codée from-scratch dans le test, indépendante
de `mcq_v5.*`.

### §3.B — Protocol compatibility

`ExperimentProtocol` + `NullCouplingOperator` préserve exactement la
dynamique géométrique pure :
- M=1, ψ constant : strictement inchangé après n steps
- M=1, pic central : identique bit-pour-bit à une boucle
  `GeometryOperator.apply + Solver.step` isolée
- M=2 NullCoupling, double vérification : m1 constant reste constant,
  m0 évolue identiquement à un m0 isolé en M=1
- historique `(k+1) × M` HistoryEntry, tous `committed=True`
- anti-aliasing temporel vérifié : `psi_step0 is not psi_stepk`
- checkpoint neutre : dynamique identique avec ou sans
  `checkpoint_interval`
- `export_protocol_metadata()` JSON-strict (`allow_nan=False`) avec
  round-trip complet

### §3.C.1 — Reference availability

Audit lecture seule des artefacts 6d. Données exactes du verdict
exporté dans `tests/v5_0b_reference_audit_result.json`.

---

## §4. Disponibilité des références 6d — état réel

Quatre fichiers JSON détectés dans `/mnt/user-data/uploads/`.
Tous sont des dicts valides. **Tous sont en schema variant** par
rapport aux clés attendues par le protocole V5-0b. Aucun fichier n'est
en `INVALID_JSON`. Aucun fichier n'est en `MISSING`.

### §4.1 — `lambda_A_v2`

- **Statut** : `AVAILABLE_SCHEMA_VARIANT`
- **Chemin** : `/mnt/user-data/uploads/6d_lambda_A_v2.json`
- **Clés top-level présentes** (10) :
  `E0`, `E1_by_time`, `E2_by_time`, `amp_P6`, `checkpoint_times`,
  `dt_simulation`, `h0`, `target_amp`, `thresholds_R_sym`,
  `verdict_global`
- **Clés attendues absentes au top-level** : `D_proj`, `R_sym`
- **Interprétation** (sans recalcul) : le fichier contient des
  trajectoires E0/E1/E2 et des seuils `thresholds_R_sym`, mais les
  agrégats `D_proj` et `R_sym` ne sont pas exposés au top-level. Ils
  sont probablement imbriqués dans `verdict_global` ou dérivés à la
  lecture. Aucune conclusion n'est tirée ici.

### §4.2 — `p5bis_A`

- **Statut** : `AVAILABLE_SCHEMA_VARIANT`
- **Chemin** : `/mnt/user-data/uploads/6d_p5bis_A.json`
- **Clés top-level présentes** (5) :
  `beta_60`, `beta_controls`, `determinism_control`, `metadata`,
  `preliminary_summary`
- **Clés attendues absentes au top-level** : `B2`, `DISSOC`, `Dh`,
  `frac_grad_active`, `frac_h_active`, `grad_status`,
  `jaccard_h_grad`
- **Interprétation** : structure organisée autour de blocs β/contrôles
  et `preliminary_summary`. Les invariants canoniques P5bis-A sont
  probablement dans `preliminary_summary` ou `beta_60`. Le schema diff
  est documenté pour usage V5-1 / V5-0b-H ultérieur.

### §4.3 — `p4_strict_A`

- **Statut** : `AVAILABLE_SCHEMA_VARIANT`
- **Chemin** : `/mnt/user-data/uploads/6d_p4_strict_A.json`
- **Clés top-level présentes** (13) :
  `R2_progression`, `alpha_progression`, `anisotropy_label`,
  `anisotropy_max_diff`, `excluded_features`, `feature_names`,
  `metadata`, `n_features_per_level`, `regression_results`,
  `reproduction_control`, `states_metadata`, `top_pairs`,
  `verdict_candidate`
- **Clés attendues absentes au top-level** : `P4-NO-STRONG-SIGNAL`,
  `variance_audit`
- **Interprétation** : structure orientée régression
  (`regression_results`, `alpha_progression`, `R2_progression`,
  `top_pairs`, `verdict_candidate`). Les agrégats `P4-NO-STRONG-SIGNAL`
  et `variance_audit` du protocole V5-0b ne correspondent pas à cette
  structure ; ils seraient dérivés via `verdict_candidate` ou
  `regression_results`. Note méthodologique : la clé attendue
  `P4-NO-STRONG-SIGNAL` correspond à une décision d'interprétation,
  pas nécessairement à un nom de champ stocké.

### §4.4 — `p4_var`

- **Statut** : `AVAILABLE_ALIAS`
- **Chemin** : `/mnt/user-data/uploads/6d_p4_var_audit.json`
- **Alias résolu** : fichier canonical `6d_p4_var.json` non trouvé,
  alias `6d_p4_var_audit.json` trouvé et accepté
- **Clés top-level présentes** (10) :
  `class_stats`, `metadata`, `modifiers`, `note_main`,
  `recommandation`, `regression_by_regime`,
  `regression_global_per_axis_level`, `significant_classes`,
  `stratifiers`, `verdict_main`
- **Clés attendues absentes au top-level** : `RR2`, `RR3`, `RSR`,
  `STR`
- **Interprétation** : audit P4-VAR avec régression par régime,
  classes stratifiées, verdict principal. Les profils canoniques
  `RR2/RR3/STR/RSR` ne sont pas exposés tels quels au top-level.

---

## §5. Frontière `FULL_REPRODUCTION = OUT_OF_SCOPE`

V5-0a contient explicitement :
- la géométrie scalaire conformal-conservative côté ψ
- l'orchestration multi-module avec couplage modulaire optionnel
- les diagnostics observationnels (héritage P5bis-A)
- l'historique committed-only

V5-0a **ne contient pas** :
- la dynamique h sédimentation/érosion 6d
- les opérateurs Ch3 (𝒞^{mod} non-trivial, novelty, etc.)
- les solveurs implicites (stubs seulement)

Tenter une reproduction complète P5/P5bis/P4 par `ExperimentProtocol`
actuel produirait des trajectoires divergentes (h figé là où le moteur
6d évoluerait). Cette divergence serait **attendue** et ne constituerait
pas un échec dynamique.

**Inscription stricte** : `FULL_6D_REPRODUCTION_OUT_OF_SCOPE`.

Le bloc `REFERENCE_6D_AVAILABLE` du §4 ci-dessus documente l'état des
artefacts oracle pour usage **ultérieur** (V5-1 ou V5-0b-H), pas pour
une comparaison numérique immédiate.

---

## §6. Anti-glissement — ce que ce rapport n'écrit pas

Le rapport ne contient et n'inscrit en aucune façon :
- "V5 reproduit P5bis"
- "V5 valide P4"
- "𝒢 confirmé en V5"
- "Ch4 atteint"
- "Validation complète du moteur 6d"
- "MCQ démontré"

Il inscrit uniquement, dans son périmètre testé :
- "V5-0a `GeometryOperator` reproduces the 6d-compatible scalar
  conformal RHS within machine precision under the tested conditions."
- "V5-0a `ExperimentProtocol` + `NullCouplingOperator` preserves the
  pure geometric dynamics with no cross-module artifact under the
  tested conditions."
- "6d reference artifacts present with schema variant; full numerical
  reproduction is out of scope without explicit h-dynamics."

---

## §7. Critères de fermeture V5-0b — checklist

| Critère | Statut |
|---|---|
| `GEOMETRY_COMPAT_PASS` strict | ✓ |
| `PROTOCOL_COMPAT_PASS` strict | ✓ |
| `FULL_REPRODUCTION` classé `OUT_OF_SCOPE` (jamais `FAIL`, jamais `PASS`) | ✓ |
| Aucun module `mcq_v5.*` modifié pour faire passer un test | ✓ |
| Document de clôture nomme les frontières de portée | ✓ |
| `REFERENCE_6D_AVAILABLE` audité (non-bloquant) | ✓ |

**V5-0b est fermée**.

---

## §8. Frontière ouverte — décision V5-0c vs V5-0b-H

À l'issue de V5-0b, les deux trajectoires définies au §8 du document
de protocole restent valides :

### Trajectoire α — V5-0c (dry coupling tests)

Stress test architectural : scheduler actif + `NullCouplingOperator`,
vérifie l'infrastructure de couplage (contexts, contributions,
aggregation) sans physique de couplage. Aucune extension de modèle.

**Coût** : faible (tests supplémentaires uniquement)
**Risque** : très bas
**Apprentissage produit** : couverture de l'infrastructure de couplage
au-delà du M=1 / M=2 NullCoupling testé en V5-0b

### Trajectoire β — V5-0b-H (h-dynamics 6d-compatible)

Extension architecturale : nouvel opérateur explicite `HDynamics`
permettant ensuite une reproduction directe P5bis-A / P4 strict via V5.

**Coût** : élevé (mini-spec, nouveau composant, intégration dans
`ExperimentProtocol`, tests bit-à-bit avec moteur 6d)
**Risque** : moyen — c'est une vraie extension, pas une compatibilité
**Apprentissage produit** : possibilité de mesurer numériquement la
compatibilité dynamique longue avec le moteur 6d

### Recommandation neutre

Le rapport ne tranche pas. Les deux trajectoires sont productives. Le
choix dépend de l'objectif immédiat :
- consolidation architecturale → α
- ouverture vers comparaison numérique 6d ↔ V5 → β

Cette décision peut être prise séparément, à un moment opportun. Le
verrouillage actuel de V5-0b ne préempte ni l'une ni l'autre.

---

## §9. Inventaire final

### Code source
```
src/mcq_v5/                                  14 modules, inchangés
```

### Tests V5-0a
```
tests/test_grid.py                           19 tests PASS
tests/test_state.py                          44 tests PASS
tests/test_views.py                          37 tests PASS
tests/test_metric_scalar.py                  36 tests PASS
tests/test_metric_tensor_stub.py             26 tests PASS
tests/test_module.py                         27 tests PASS
tests/test_instance.py                       29 tests PASS
tests/test_geometry_operator.py              48 tests PASS
tests/test_coupling_context.py               43 tests PASS
tests/test_coupling_scheduler.py             53 tests PASS
tests/test_coupling_operator.py              41 tests PASS
tests/test_coupling_aggregator.py            41 tests PASS
tests/test_solver.py                         40 tests PASS
tests/test_diagnostics.py                    57 tests PASS
tests/test_experiment.py                     41 tests PASS
```

### Tests V5-0b
```
tests/test_v5_0b_geometry_compatibility.py   15 tests PASS
tests/test_v5_0b_protocol_compatibility.py   16 tests PASS
tests/test_v5_0b_reference_json_availability.py
                                             10 tests PASS
```

### Documentation
```
docs/V5_0_architecture_spec.md
docs/V5_0a_interfaces_plan.md
docs/V5_0b_6d_compatibility_protocol.md      (corrigé §3.A.3)
docs/V5_0b_closure_report.md                 (ce document)
```

### Artefacts exportés
```
tests/v5_0b_reference_audit_result.json      (verdict §3.C.1 sérialisé)
```

---

*Fin du rapport de clôture V5-0b.*
