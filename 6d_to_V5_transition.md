# 6d → V5 — Document de transition

**Statut** : document compagnon autonome. Pas un § du cadrage 6d.
Pas un nouveau cycle expérimental. Pas un cahier des charges V5
au sens d'engagement de programme.

§23.6 du cadrage 6d avait explicitement préservé V5 hors du
cadrage. Ce document maintient cette séparation : le cadrage 6d
reste fermé sur lui-même avec ses amendements §11 → §25.8. Le
document de transition est extérieur à cette histoire.

**Objet** : acter la clôture du programme 6d comme **constat de
maturité**, identifier les dettes restantes comme dettes
**architecturales** (et non d'instrumentation), et nommer les
trajectoires possibles vers V5 **sans trancher**.

**Garde-fou** : pas de Δ, 𝒢, RR/RR², MCQ comme cible, Ch4 immédiat,
ni engagement V5.

---

## 1. Clôture comme maturité, pas comme épuisement

Le programme 6d-α à 6d-λ, et les inscriptions §11 à §25.8 du
cadrage, ont atteint un point où **les questions encore ouvertes
ne sont plus des dettes du moteur lui-même**.

Les dettes structurantes du cadrage initial ont toutes reçu un
test direct sous leur forme propre :
- propriétés Ch3 P1/P2/P3 : §13 (corridor), §16/§17 (axe G/A
  projectif), §18 (axe non compositionnel), §19 (interaction h×P′
  faible)
- contrainte temporelle ψ→h : §21 (latence inverse)
- séparabilité §1.8 : §22 (test direct λ-A-v2)
- multi-bassins et dissociation fonctionnelle : §24 (P5bis-A,
  DISSOC-BASIN-SPECIFIC sur B2)
- reconstructibilité limitée : §25 (P4-A NO-STRONG-SIGNAL),
  §25.8 (P4-VAR STRATIFIED)

Aucune dette interne du moteur n'est restée non testée.

Les questions encore ouvertes après §25.8 sont :
- couplage inter-modulaire 𝒞^{mod} natif (Ch3)
- multi-instance 𝒞^N (MCQᴺ)
- tensorialisation H(θ) (Ch4)
- opérateurs géométriques au-delà du conformal-conservatif

Aucune de ces questions n'est testable dans le moteur 6d en
l'état. Toutes exigent une **structure nouvelle**, pas un cycle
nouveau.

C'est précisément ce signal qu'un programme expérimental a atteint
son terme naturel. La clôture n'est pas une fuite ; elle est la
reconnaissance que les outils 6d ont produit ce qu'ils pouvaient
produire.

---

## 2. Limites observées en 6d

### 2.1 Limites structurelles déjà inscrites (§23.5 et §25.7)

- grille 5×5×5 très coarse
- N = 1 (instance unique, hors MCQᴺ)
- approximation conformal-conservative, pas Laplace-Beltrami strict
- h(θ) scalaire conforme, pas H(θ) tensoriel
- phénomènes fins d'anisotropie, non-commutativité, latence locale,
  transport parallèle non testés proprement ou hors portée
  instrumentale

### 2.2 Dette architecturale révélée — 𝒞^{mod}

Le formalisme Ch3 inscrit la loi de mouvement sous la forme :

  ∂t ψᵢ^α = (diffusion géométrique) + (drift géométrique)
          + 𝒞ᵢ^{mod,α} + 𝒞ᵢ^{N,α} + ηᵢ^α

En N = 1, 𝒞ᵢ^N disparaît, **mais 𝒞ᵢ^{mod} reste**. Il est prévu
comme couplage inter-modulaire non-gradient, irréductible à un
drift dans un potentiel effectif Φ_eff.

Le moteur 6d implémente uniquement la diffusion géométrique +
h-dynamics + perturbations bornées sur un seul module ψ. Il ne
contient **ni plusieurs modules**, ni recouvrements R_ij entre
modules, ni évaluation de nouveauté dans la métrique propre d'un
autre module, ni couplage non-gradient ajouté à ∂t ψ.

**6d n'a ni validé ni invalidé 𝒞^{mod}. Le moteur ne possédait
pas les degrés de liberté requis pour l'instrumenter.**

Cette dette n'est pas une dette d'instrumentation laissée ouverte
par 6d. C'est une dette **architecturale** révélée par les limites
de portée du cadrage initial. Son test propre exige une architecture
nouvelle.

### 2.3 Statut épistémique des limites

Les limites de §2.1 et la dette de §2.2 sont **observées**, pas
postulées. Elles encadrent ce que les cycles §13-§25 ont pu
produire, et elles indiquent ce qu'un toy model ultérieur devrait
être conçu pour adresser.

Elles ne valent pas comme preuves de nécessité. Un cadre théorique
alternatif pourrait choisir d'autres limites. Mais à l'intérieur
du programme MCQ tel qu'il est instrumenté, ces limites sont
les frontières actuelles de l'observable.

---

## 3. Critères de conception pour V5 (sans engagement)

Les critères suivants sont des **réponses possibles aux dettes
observées**, pas un engagement d'implémentation. V5 reste un
horizon, pas un programme inscrit.

### 3.1 Résolution / support

V5 ne devrait pas se contenter d'augmenter la grille. Il devrait
réduire l'effet passe-bas du 5×5×5 :
- grille plus fine ou support adaptatif
- capacité à observer des poches locales, gradients fins,
  frontières de bassin
- métriques de variance locale du bruit, héritées de P4-VAR
  (§25.8)

### 3.2 Couplage modulaire natif 𝒞^{mod}

V5 devrait implémenter 𝒞^{mod} comme terme autonome dans ∂t ψ,
pas comme drift caché dans Φ_eff :
- plusieurs modules ψᵢ
- métriques hᵢ ou Hᵢ propres
- recouvrements R_ij
- nouveauté de j évaluée dans la métrique de j
- comparaison à la forme de i dans la métrique de i
- couplage maximal à recouvrement intermédiaire
- couplage nul à R → 0 et R → 1

### 3.3 Non-gradient / circulation

V5 devrait pouvoir produire des composantes de flux non réductibles
à −∇Φ :
- mesurer curl J ou circulation discrète
- séparer flux gradient vs flux coupling
- tester si 𝒞^{mod} crée une circulation non irrotationnelle
- ne pas cacher 𝒞^{mod} dans Φ_eff

### 3.4 Multi-instance en option

𝒞^N peut rester un V5-bis si trop lourd, mais V5 devrait être
conçu pour ne pas empêcher son ajout. Le mono-instance avec
multi-modules est le minimum architectural pour adresser
𝒞^{mod} ; le multi-instance natif serait la version complète
MCQᴺ.

### 3.5 Géométrie : deux trajectoires

**Avertissement de non-séparabilité (𝒞^{mod} ⊥ H(θ))** : La
disjonction opérationnelle entre V5a et V5b procède d'une
hypothèse de travail commode mais non démontrée d'orthogonalité
des défaillances. Par conséquent, les mentions de « dette
principale » ci-dessous sont rétrogradées au statut d'« axe
d'attaque focal ». Si le couplage non-gradient natif (𝒞^{mod})
et la non-séparabilité de la métrique (H(θ)) sont les
manifestations indissociables d'une même structure sous-jacente,
l'isolement artificiel d'un axe s'expose à une réplication stérile
des artefacts de lissage et des dégénérescences de type B3
constatés en 6d.

Deux réponses à des dettes différentes, **présentées sans choix
préalable** :

**V5a — axe d'attaque focal : couplage modulaire**
- géométrie encore scalaire h(θ)
- multi-module natif
- 𝒞^{mod} comme opérateur autonome
- séparation flux gradient / flux coupling
- objectif : mesurer ce qu'apporte le couplage avant toute
  tensorialisation, isoler l'effet 𝒞^{mod} du plafond scalaire
  observé en 6d (**sous réserve que cet isolement soit
  effectivement possible** — cf. avertissement ci-dessus)

**V5b — axe d'attaque focal : géométrie tensorielle**
- H(θ) tensoriel pleinement instrumenté
- ouverture de l'espace compatible Ch4
- opérateurs au-delà du conformal-conservatif (Laplace-Beltrami
  strict, transport parallèle)
- objectif : adresser directement la dette géométrique Ch4
  (**sans présupposer que cette dette est indépendante du
  couplage modulaire** — cf. avertissement ci-dessus)

La décision entre V5a et V5b dépendra autant de considérations
méthodologiques que pratiques. Le présent document n'instruit
pas ce choix. Une décision ultérieure pourra être :
- V5a d'abord, V5b ensuite (isoler couplage avant tensorialisation)
- V5b direct (adresser la dette la plus structurante)
- une combinaison V5a + V5b si l'architecture le permet sans
  confusion des effets

---

## 4. Ce que V5 ne doit pas perdre

V5, sous quelque forme qu'il prenne, n'est pas un redépart de
zéro. Il doit hériter des instruments qui ont **réellement produit
de l'information** dans 6d :

- diagnostics de bassins (§24, P5bis-A)
- diagnostics de dissociation fonctionnelle (§22, §24)
- audits de variance (P4-VAR, §25.8)
- tests de réactivation (P5bis-A, avec caveat méthodologique
  §24.6 sur la sensibilité au seuil relatif)
- séparation stricte entre faits empiriques et interprétation
- reproductibilité déterministe stricte (vérifiée P5bis-A, P4-A)
- contrôles d'underflow et diagnostics de portée
- documents compagnons avant code (κ-synthèse, λ-0, P5bis-0, P4-0)
- format de cadrage avec amendements traçables

V5 doit hériter non seulement des succès de 6d, mais aussi de
ses échecs documentés :
- λ-A v0 rejeté méthodologiquement (métriques opératoires non
  robustes aux cellules quasi-effondrées)
- métrique de réactivation fonctionnelle non robuste au seuil
  relatif
- verdict P4 automatique heuristique rejeté en lecture
- nécessité de séparer intra-trajectoire / inter-bassin avant
  toute régression globale

Ces leçons méthodologiques font partie de l'héritage du programme.

---

## 5. Ce qui n'est pas réglé par la transition

Le présent document n'est pas un programme V5. Il n'engage rien.
Il acte un constat de maturité 6d et identifie les directions
possibles. Les décisions suivantes restent ouvertes :
- ouvrir V5 ou reporter ?
- si V5, V5a ou V5b ou les deux ?
- avec quelle équipe / quel calendrier ?
- avec quelle continuité avec Ch4 empirico-théorique ?

Aucune de ces questions ne relève du cadre 6d. Elles relèvent
d'une décision de programme externe, à l'image du parallèle
§22 / §23 (verdict empirique séparé de la décision de programme).

---

## 6. Formulation finale

> La branche 6d scalaire est close comme programme d'exploration
> du moteur conformal-conservatif instrumenté. Elle n'est pas une
> réfutation de 𝒞^{mod}, ni un test du couplage non-gradient
> natif, ni une falsification de MCQ. Elle a cartographié les
> limites de ce que h(θ) scalaire sous approximation
> conformal-conservative peut produire comme structure de
> réalisabilité, et elle a documenté ces limites avec leurs
> caveats. **Le couplage modulaire 𝒞^{mod} et la géométrie
> tensorielle H(θ) révèlent deux dettes architecturales dont la
> séparabilité reste elle-même une hypothèse à tester ; ils
> définissent les lignes de rupture potentielles d'un V5 ultérieur,
> sans qu'aucune trajectoire ne soit engagée par cette clôture.**
> Tout résultat 6d reste relatif à cette portée.

---

## 7. Garde-fou final

- 6d n'a ni validé ni invalidé 𝒞^{mod}.
- 6d n'a ni validé ni invalidé la self-opacity Ch3 §3.5.
- 6d n'a ni validé ni invalidé MCQ.
- 6d a cartographié la réalisabilité h(θ) scalaire
  conformal-conservative, sous la grille 5×5×5, N = 1, et le
  protocole de perturbations bornées documenté §11.
- Tout résultat 6d reste **relatif à cette portée**.
- **Refus de la coordination disjonctive** : La transition ne
  présuppose pas que 𝒞^{mod} et H(θ) soient des variables
  indépendantes ou des choix exclusifs. Il demeure conceptuellement
  plausible qu'elles constituent les deux faces d'une seule et
  même dette géométrique non-séparable. Tout programme V5 ultérieur
  qui ferait l'économie de cette indétermination s'exposerait à
  une perte de description majeure par réduction méthodologique
  prématurée.
- Le présent document n'engage ni V5a, ni V5b, ni un calendrier.
- La coda empirique du cadrage 6d (§22 / §24) reste intacte et
  ne s'étend pas à V5.

---

*Fin du document de transition. Le cadrage 6d reste fermé sur
lui-même. V5 reste suspendu. Conformal-conservatif ou rien
reste la règle 6d. Si V5 est ouvert un jour, ce sera sous une
règle nouvelle, à inscrire dans un cadrage V5 propre.*
