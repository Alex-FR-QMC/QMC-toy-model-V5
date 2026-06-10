# V5-0b — Protocole de compatibilité 6d

**Statut** : document compagnon, à valider avant tout code.
**Préalable** : V5-0a fermée (étapes 1–14 + guard aggregator/grid),
582/582 tests PASS.
**Posture** : V5-0b est un **harnais d'audit**, pas une étape de
nouvelle physique. Aucun module V5-0a ne sera modifié, à l'exception
éventuelle d'un test transversal.

---

## §1. Pourquoi un document avant code

Lancer V5-0b comme "reproduction immédiate des résultats P5-direct /
P5bis / P4 strict" serait une erreur de cadrage. V5-0a contient
explicitement :

- la géométrie scalaire conformal-conservative côté ψ
  (`flux = -h_face·grad(ψ)`, `divergence` = `-div(J)`)
- l'orchestration multi-module ψ + couplage modulaire optionnel
- l'historique et les diagnostics observationnels

V5-0a **ne contient pas** explicitement :

- la dynamique h sédimentation/érosion du moteur 6d
- les opérateurs étendus de Ch3 (𝒞^{mod} non-trivial, novelty, etc.)
- les solveurs implicites

Donc la **non-reproduction** des résultats P5bis-A et P4 strict
**est attendue**. Confondre cette non-reproduction avec un échec
dynamique serait inscrire une fausse défaite. La distinction doit
être méthodologique avant d'être numérique.

---

## §2. Critère étape V5-0b

> V5-0b validates that the V5-0a architecture reproduces the
> 6d-compatible scalar conformal geometry and protocol behavior under
> null coupling, while explicitly classifying long-horizon P5/P5bis/P4
> reproduction as reference-only or out-of-scope unless a 6d h-dynamics
> operator is specified.

---

## §3. Stratification — trois niveaux

V5-0b produit **plusieurs verdicts distincts**, jamais un verdict global :

```
GEOMETRY_COMPAT_PASS / FAIL
PROTOCOL_COMPAT_PASS / FAIL
REFERENCE_6D_AVAILABLE / MISSING
FULL_6D_REPRODUCTION_OUT_OF_SCOPE / PASS / FAIL
```

### §3.A — Niveau A : compatibilité opérateur géométrique

**Objectif** : vérifier que `GeometryOperator` V5-0a reproduit le cœur
spatial 6d :

```
h_face = harmonic_mean(h_left, h_right)
J      = -h_face · grad(ψ)
dψ/dt  = -div(J)
Neumann zero-flux aux bords externes
```

**Tests minimaux** (one-step RHS comparison) :
- A.1 — champ constant → RHS = 0 partout
- A.2 — pic central [0,1,0] → RHS = [1, −2, 1] (verrouillé en V5-0a)
- A.3 — champ linéaire selon un axe sous Neumann zero-flux :
  RHS nul sur l'**intérieur**, contributions opposées aux **deux bords**,
  conservation globale `sum(RHS) = 0`.
  Exemple 1D avec `ψ=[0,1,2,3,4]`, `h=1`, `dx=1` :
  `grad = [1,1,1,1]`, `J = [-1,-1,-1,-1]`, `RHS = [1,0,0,0,-1]`,
  `sum(RHS) = 0`. Ne pas tester "RHS = 0 partout" — ce serait faux
  sous la convention Neumann implicite.
- A.4 — champ aléatoire déterministe seedé (`np.random.default_rng(42)`)
  → comparer avec une fonction de référence 6d codée localement dans
  le test (pas dans `mcq_v5.*`). **Cas en 3D obligatoirement** pour
  garantir la compatibilité dimensionnelle.
- A.5 — h variable déterministe → vérifier que la moyenne harmonique
  des faces produit les résultats 6d attendus
- A.6 — conservation : `|sum(dψ)| < 1e-12` pour tous les cas ci-dessus

**Tolérance** : `atol = 1e-12` pour one-step RHS, par cellule.

**Verdict attendu** : `GEOMETRY_COMPAT_PASS` strict. Si FAIL,
inscription explicite du delta cellule par cellule pour audit.

### §3.B — Niveau B : compatibilité protocolaire minimale

**Objectif** : vérifier que `ExperimentProtocol` + `NullCouplingOperator`
ne change pas la dynamique géométrique (couplage neutre ≡ géométrie pure).

**Tests minimaux** :
- B.1 — M=1, ψ constant → état strictement inchangé après n steps
  (déjà testé en V5-0a, à confirmer dans le harnais V5-0b)
- B.2 — M=1, pic central → diffusion identique à
  `GeometryOperator.apply` appliqué dt fois
- B.3 — M=2, NullCoupling, **double vérification** :
  (i) avec m0=pic et m1=constant, m1 reste strictement constant
      (couplage nul, aucun effet croisé) ;
  (ii) m0 évolue exactement comme un m0 isolé dans une Instance M=1
      (l'orchestration multi-module ne bloque ni n'altère la dynamique
      locale).
- B.4 — historique cohérent : `(k+1) × M` HistoryEntry après k steps
  si `commit_initial=True`
- B.5 — checkpoint neutre : avec ou sans `checkpoint_interval`, la
  dynamique est identique (seules les métadonnées diffèrent)
- B.6 — métadonnées exportables JSON strict : `json.dumps(bundle,
  allow_nan=False)` succès sur `export_protocol_metadata()`

**Verdict attendu** : `PROTOCOL_COMPAT_PASS` strict.

### §3.C — Niveau C : compatibilité résultats 6d longs

**Distinction obligatoire** entre deux sous-cas :

#### §3.C.1 — Reproduction par oracle 6d (référence statique)

**Objectif** : vérifier que les artefacts 6d de référence sont stables
et disponibles. Lecture seule, aucun calcul V5.

**Fichiers attendus** (s'ils existent) :
```
6d_lambda_A_v2.json      — résultats P5-direct λ-A-v2
6d_p5bis_A.json          — résultats P5bis-A (DISSOC, B2, Dh, etc.)
6d_p4_strict_A.json      — résultats P4 strict
6d_p4_var.json           — résultats P4-VAR
```

**Tests minimaux** :
- C.1.1 — chargement JSON (succès / échec)
- C.1.2 — présence des clés critiques :
  - λ-A-v2 : `D_proj`, `R_sym`
  - P5bis-A : `DISSOC`, `B2`, `Dh`, `frac_h_active`,
    `frac_grad_active`, `jaccard_h_grad`, `grad_status`
  - P4 strict : `P4-NO-STRONG-SIGNAL`, `variance_audit`
  - P4-VAR : `RR2`, `RR3`, `STR`/`RSR` profils

**Verdict** : `REFERENCE_6D_AVAILABLE` ou `MISSING` (par fichier).

Pas de calcul, pas de comparaison. Juste disponibilité.

**Note** : les fichiers ne sont pas garantis existants au moment où
V5-0b sera codé. Le harnais doit gérer l'absence gracieusement
(`MISSING` ≠ `FAIL`).

#### §3.C.2 — Reproduction par V5 (calcul complet)

**Sub-décision méthodologique** :

V5-0a ne contient pas la dynamique h sed/ero 6d. Toute tentative de
relancer P5bis ou P4 strict via `ExperimentProtocol` actuel produira
des résultats divergents (h restera figé là où le moteur 6d ferait
évoluer h via sédimentation/érosion).

**Classification stricte** :

```
FULL_6D_REPRODUCTION_OUT_OF_SCOPE
```

et **non** :

```
FULL_6D_REPRODUCTION_FAIL
```

Inscrire `FAIL` ici reviendrait à confondre **absence d'implémentation**
avec **échec dynamique**. La distinction doit être protégée.

---

## §4. Décision méthodologique à valider

Avant tout code V5-0b, il faut trancher entre deux options :

### Option 1 — V5-0b strict minimal (recommandée)

V5-0b = harnais de compatibilité **sans** nouvelle physique.

Couverture :
- §3.A — Geometry compatibility (calcul)
- §3.B — Protocol compatibility (calcul)
- §3.C.1 — Reference availability (lecture seule)

Ce que V5-0b déclare explicitement comme hors portée :
- §3.C.2 — full P5bis / P4 strict reproduction
  → status `BLOCKED_BY_MISSING_H_DYNAMICS`

**Avantages** :
- respecte V5-0a (aucune nouvelle physique furtive)
- pas de drift méthodologique (compatibilité ≠ extension)
- documente proprement la dette de modèle
- prépare le terrain pour V5-0c ou V5-0b-H sans précipitation

### Option 2 — V5-0b-H (ajouter `HDynamicsOperator`)

Étape distincte, **plus lourde**. Si choisie, exige :
- mini-spec préalable : `qui calcule dh/dt ?`, `où dans
  ExperimentProtocol ?`, `comment additionner au RHS State ?`
- nouveau test transversal : compatibilité bit-à-bit avec moteur 6d
- intégration dans `Module` ou dans un opérateur séparé
- gestion des champs solver_fields étendus

**Inconvénient principal** : on ne fait pas de la compatibilité, on
fait de l'extension. Le risque est de mêler les deux postures, ce qui
défait la rigueur posée depuis le cadrage 6d.

**Recommandation** : reporter à V5-0c ou V5-0b-H **après** confirmation
par V5-0b que le cœur géométrique est compatible.

---

## §5. Recommandation finale

Choisir **Option 1** : V5-0b = harnais strict minimal.

Trois fichiers à créer (jamais sous `mcq_v5.*`) :
```
docs/V5_0b_6d_compatibility_protocol.md   ← ce document
tests/test_v5_0b_geometry_compatibility.py
tests/test_v5_0b_protocol_compatibility.py
tests/test_v5_0b_reference_json_availability.py   ← optionnel
```

Pas de modification des modules V5-0a.

V5-0b se ferme avec un **rapport stratifié** :

```
GEOMETRY_COMPAT     : PASS / FAIL
PROTOCOL_COMPAT     : PASS / FAIL
REFERENCE_AVAILABLE : per-file PASS / MISSING
FULL_REPRODUCTION   : OUT_OF_SCOPE (h-dynamics not in V5-0a)
```

---

## §6. Anti-glissement — formules verrouillées

V5-0b **n'écrit pas** dans son rapport :

- "V5 reproduit P5bis"
- "V5 prouve la compatibilité MCQ"
- "𝒢 confirmé en V5"
- "Ch4 atteint"
- "Validation complète du moteur 6d"

V5-0b **peut écrire** :

- "V5-0a geometry operator reproduces 6d-compatible RHS within
  machine precision under the tested conditions"
- "Full P5bis/P4 reproduction requires an explicit h-dynamics operator
  not present in V5-0a; therefore classified as OUT_OF_SCOPE"
- "Reference 6d artifacts available / missing on disk"

---

## §7. Critères de sortie V5-0b

V5-0b est considérée **validée** si et seulement si :

1. `GEOMETRY_COMPAT_PASS` strict (cellule par cellule, `atol=1e-12`)
2. `PROTOCOL_COMPAT_PASS` strict (sur tous les sous-tests §3.B)
3. Le rapport `FULL_REPRODUCTION` est **explicitement** classé
   `OUT_OF_SCOPE`, jamais `FAIL`, jamais `PASS`
4. Aucun module `mcq_v5.*` n'a été modifié pour faire passer un test
5. Le document de clôture V5-0b nomme les frontières de portée

---

## §8. Frontière ouverte : V5-0c vs V5-2

À l'issue de V5-0b, **deux trajectoires** restent possibles :

### Trajectoire α — V5-0c = dry coupling tests

Tests avec scheduler actif + `NullCouplingOperator`, vérifiant que
l'**infrastructure de couplage** (contexts, contributions, aggregation)
fonctionne, sans physique de couplage.

C'est essentiellement un **stress test architectural**, sans extension
de modèle.

### Trajectoire β — V5-0b-H = h-dynamics 6d-compatible

Ajout d'un opérateur explicite de dynamique h. Permet ensuite de
tenter §3.C.2. C'est une **vraie extension** architecturale.

**Décision déférée** au verdict de V5-0b. Le présent document ne
tranche pas. La tension entre ces deux trajectoires reste
**productive et ouverte** :
- α renforce la rigueur architecturale sans nouvelle physique
- β ouvre la possibilité de mesurer la compatibilité dynamique longue

Inscrire l'une ou l'autre comme "déjà décidée" reviendrait à
court-circuiter l'apprentissage que V5-0b doit produire.

---

## §9. Inscription finale du périmètre V5-0a

Avant de quitter V5-0a, inscription formelle :

**V5-0a fermée** :
- 14 étapes complétées
- 582/582 tests PASS
- guards `geometry.shape/dx` et `aggregator.geometry.shape/dx`
  appliqués
- anti-centralisation vérifiée à chaque maillon
- documentation présente : `V5_0_architecture_spec.md`,
  `V5_0a_interfaces_plan.md`, et le présent document

**V5-0a contient** :
- géométrie scalaire conformal-conservative ψ
- chaîne complète d'orchestration (multi-module avec couplage optionnel)
- diagnostics observationnels
- historique committed-only

**V5-0a ne contient pas** :
- dynamique h sédimentation/érosion 6d
- opérateurs Ch3 (𝒞^{mod} non-trivial, novelty transmitted, etc.)
- solveurs implicites (stubs uniquement)
- intégration tensorielle (stub TensorMetric uniquement)

Ces absences ne sont **pas des dettes architecturales**. Elles sont
des **choix de portée** explicitement déclarés. La portée future
(V5-0b, V5-0c, V5-0b-H, V5-1, V5-2) en hérite.

---

*Fin du document.*
