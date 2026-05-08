# Journal de projet

Ce document sert de trace continue du projet. Il sera mis à jour à chaque étape importante pour préparer le rapport final.

## Description du projet

### Contexte

Le projet consiste à classifier un grand catalogue de produits dans la taxonomie Google Products.

Les données produit disponibles sont :

- `hashed_external_id`
- `title`
- `description`
- `brand`
- `sale_price`

Le vrai dataset principal contient actuellement :

- `128253` produits dans `data/ensae_export_without_l1.parquet`
- `128253` labels `level_1_name` dans `data/ground_truth_level_1.parquet`

La taxonomie cible est stockée dans `taxonomy.txt`. Elle contient :

- `34` nœuds de niveau 1
- `191` nœuds de niveau 2
- `5440` nœuds au total
- une profondeur maximale observée de `7`

Répartition par profondeur :

- profondeur 1 : `34`
- profondeur 2 : `191`
- profondeur 3 : `1330`
- profondeur 4 : `2142`
- profondeur 5 : `1323`
- profondeur 6 : `378`
- profondeur 7 : `42`

### Objectif métier

L'objectif est d'assigner chaque produit à une catégorie feuille de la taxonomie Google.

Contraintes importantes :

- le `level 1` dispose d'un ground truth
- les niveaux `2+` n'ont pas de ground truth complet
- l'inférence doit rester rapide
- la solution doit pouvoir se généraliser à grande échelle

### Stratégie actuelle du projet

La stratégie retenue à ce stade est hiérarchique :

1. `Level 1`
   Un modèle d'embeddings fine-tuné supervisé est déjà utilisé avec de bonnes performances.

2. `Level 2+`
   On n'essaie pas de tout résoudre avec un gros classifieur supervisé sans labels réels.
   La stratégie active est maintenant :
   - multi-prototypes textuels par catégorie
   - embeddings pré-calculés des prototypes
   - scoring local sur les enfants valides uniquement
   - beam search borné en cas d'ambiguïté
   - comparaison future avec une baseline simple type barycentre

### Décision de cadrage actuelle

La piste "dataset synthétique comme axe principal" est mise de côté.

Le dataset synthétique déjà généré reste disponible comme artefact de travail, mais il n'est plus au centre de la méthode.

Le focus actuel est :

- valider l'approche hiérarchique par prototypes
- annoter un petit lot réel `L2`
- mesurer la qualité réelle sur des produits annotés

### Données et artefacts actuellement disponibles

- notebook exploratoire :
  - `LV1.ipynb`
  - `LV2.ipynb`
- preprocessing modulaire :
  - `src/preprocessing.py`
- fine-tuning embeddings `L1` :
  - `src/triplet_finetuning.py`
- annotation réelle `L2` :
  - `scripts/annotate_l2_dataset.py`
- recherche hiérarchique multi-prototypes :
  - `src/hierarchical_prototype_search.py`

### État actuel du projet

Ce qui est déjà en place :

- preprocessing des produits hors notebook
- fine-tuning embeddings pour `level 1`
- export d'un lot réel à annoter pour `L2`
- outil terminal d'annotation
- implémentation d'une méthode hiérarchique `multi-prototypes + beam search`

Ce qui n'est pas encore validé expérimentalement :

- qualité réelle de la prédiction `L2` sur un lot annoté
- comparaison entre :
  - barycentre simple
  - multi-prototypes
  - multi-prototypes + beam search
- meilleur modèle d'embeddings pour `L2+`

## Résumé de reprise

### Où en est le projet

Le projet n'est plus au stade notebook-only. Il possède maintenant une base `src/` exploitable avec :

- preprocessing
- fine-tuning `L1`
- annotation manuelle `L2`
- moteur hiérarchique multi-prototypes

Le point de blocage principal n'est plus l'implémentation technique de base, mais la validation expérimentale sur du vrai `L2`.

### Fichiers les plus importants pour reprendre rapidement

- journal global :
  - `rapport/journal_projet.md`
- échantillon réel à annoter :
  - `data/real_l2_annotation_sample_llm_targeted.csv`
- outil d'annotation :
  - `scripts/annotate_l2_dataset.py`
- build de l'index multi-prototypes :
  - `scripts/build_hierarchical_prototype_index.py`
- prédiction hiérarchique :
  - `scripts/predict_with_hierarchical_prototypes.py`

### Prochaines étapes prioritaires

1. Annoter un premier lot réel `L2`

Commande recommandée :

```bash
python scripts/annotate_l2_dataset.py \
  --input-path data/real_l2_annotation_sample_llm_targeted.csv \
  --level1 "animals & pet supplies"
```

2. Valider le dataset synthétique ciblé
2. Construire l'index multi-prototypes complet

```bash
python scripts/build_hierarchical_prototype_index.py \
  --output-dir artifacts/hierarchical_prototype_index
```

3. Lancer les premières prédictions hiérarchiques

```bash
python scripts/predict_with_hierarchical_prototypes.py \
  --input-path data/preprocessed_lv2.parquet \
  --input-format parquet \
  --index-dir artifacts/hierarchical_prototype_index \
  --output-path data/hierarchical_predictions.csv \
  --level1-col level_1_name
```

4. Comparer les approches

À mesurer sur un lot annoté :

- baseline catégorie seule
- barycentre simple
- multi-prototypes
- multi-prototypes + beam search

### Questions ouvertes

- Quel modèle d'embeddings est le meilleur pour `L2+` :
  - `prdev/mini-gte`
  - `BAAI/bge-small-en-v1.5`
  - autre modèle léger

- Quelle règle de score marche le mieux :
  - `max`
  - `mean_top_k`

- Quel seuil d'ambiguïté utiliser pour déclencher le beam search

### Conseil de reprise

Si le projet doit être repris rapidement, il faut commencer par :

1. annoter un seul nœud `L1` important
2. construire l'index multi-prototypes
3. tester la prédiction hiérarchique sur ce même nœud
4. comparer à une baseline barycentre simple

Le nœud recommandé pour démarrer est :

- `animals & pet supplies`

Il est petit, simple, et permet de valider vite la mécanique `L2`.

## Format de suivi

Chaque entrée contient :

- objectif de l'étape
- changements réalisés
- fichiers créés ou modifiés
- résultats produits
- limites identifiées
- prochaine étape

## 2026-04-28 - Refactorisation du notebook LV2

### Objectif

Sortir la logique du notebook `LV2.ipynb` vers des modules Python autonomes dans `src/` et des scripts d'exécution dans `scripts/`.

### Changements réalisés

- extraction du preprocessing catalogue vers un module dédié
- extraction du fine-tuning d'embeddings avec `BatchHardTripletLoss`
- ajout de scripts CLI pour lancer preprocessing et entraînement sans notebook
- mise à jour légère du `README`

### Fichiers concernés

- `src/preprocessing.py`
- `src/triplet_finetuning.py`
- `scripts/preprocess_catalog.py`
- `scripts/train_triplet_model.py`
- `README.md`

### Résultats

- pipeline preprocessing réutilisable
- pipeline de fine-tuning configurable par hyperparamètres
- base de code plus propre que le notebook exploratoire

### Limites

- la partie hiérarchique `L2+` n'était pas encore structurée
- pas encore de workflow de génération/validation synthétique

### Prochaine étape

Ajouter une brique de validation pour les futurs datasets synthétiques `L2`.

## 2026-04-28 - Validation du dataset synthétique L2

### Objectif

Mettre en place les trois vérifications discutées pour évaluer la qualité d'un dataset synthétique `L2`.

### Changements réalisés

- ajout d'un module de validation centré sur texte + `TF-IDF + LogisticRegression`
- ajout d'une CLI pour lancer séparément ou ensemble les trois étapes
- correction des imports des scripts existants pour qu'ils fonctionnent depuis `scripts/`

### Vérifications implémentées

1. `real vs synthetic`
2. `sibling classification`
3. `synthetic train -> real test`

### Fichiers concernés

- `src/synthetic_validation.py`
- `scripts/validate_synthetic_dataset.py`
- `scripts/preprocess_catalog.py`
- `scripts/train_triplet_model.py`
- `src/__init__.py`

### Résultats

- pipeline de validation exécutable en ligne de commande
- métriques disponibles : `AUC`, `accuracy`, `macro_f1`, `classification_report`
- smoke test réalisé sur un jeu jouet pour vérifier le comportement des trois étapes

### Limites

- l'étape `synthetic -> real` suppose un vrai petit dataset annoté `L2`
- la validation de surface ne suffit pas à elle seule pour juger la qualité métier

### Prochaine étape

Produire un premier dataset synthétique bootstrap `L2` et un échantillon réel prêt à annoter.

## 2026-04-28 - Génération bootstrap du dataset synthétique L2

### Objectif

Générer un premier dataset synthétique `L2` pour tous les nœuds `L2` de la taxonomie, en se basant sur le vrai dataset `L1`, puis exporter un sous-ensemble réel à annoter.

### Méthode de génération

La génération actuelle n'est pas encore une génération LLM libre. C'est un bootstrap contrôlé, construit à partir de trois sources :

1. la structure réelle de la taxonomie `L1 -> L2` dans `taxonomy.txt`
2. les descriptions sémantiques `L2` dans `categories_level_2.csv`
3. le style observé dans le vrai catalogue pour chaque `level_1_name`

#### Détail du processus

Pour chaque catégorie `L1` :

- on récupère les vrais produits du catalogue appartenant à ce `L1`
- on construit un profil de style :
  - marques observées
  - distribution de prix
  - longueur moyenne des titres
  - longueur moyenne des descriptions

Pour chaque enfant `L2` de ce `L1` :

- on récupère la description sémantique associée dans `categories_level_2.csv`
- on extrait des termes/phrases clés depuis cette description
- on génère plusieurs couples `title + description` via des templates
- on échantillonne une marque et un prix à partir du style réel du `L1`
- on reconstruit aussi une colonne `text` au format utilisé dans le projet

Le résultat est un dataset synthétique de démarrage, dont le but est :

- d'initialiser les expériences `L2`
- d'alimenter la validation
- de servir de base avant une génération LLM plus riche et plus réaliste

### Changements réalisés

- ajout d'un module de génération bootstrap `L2`
- ajout d'une CLI de génération
- génération d'un export synthétique global
- génération d'un export réel d'annotation `L2`
- génération d'un fichier résumé avec les comptes produits

### Fichiers concernés

- `src/synthetic_generation.py`
- `scripts/generate_synthetic_l2_dataset.py`
- `src/__init__.py`
- `data/synthetic_l2_seed.csv`
- `data/real_l2_annotation_sample.csv`
- `data/synthetic_l2_generation_summary.json`

### Résultats

- `191` catégories `L2` couvertes
- `1528` lignes synthétiques générées
- `298` lignes réelles exportées pour annotation manuelle

### Limites

- ce dataset est un seed bootstrap, pas encore un dataset LLM final
- certaines catégories restent trop génériques
- le réalisme lexical dépend encore fortement des descriptions `L2` disponibles
- des marques peuvent rester imparfaites sur certaines branches

### Prochaine étape

1. annoter une partie de `data/real_l2_annotation_sample.csv`
2. lancer `scripts/validate_synthetic_dataset.py`
3. améliorer la génération catégorie par catégorie avec génération LLM ciblée sur les nœuds les plus importants

## 2026-04-28 - Génération ciblée LLM-like pour toutes les catégories L2

### Objectif

Passer d'un seed bootstrap assez générique à une génération plus ciblée par catégorie `L2`, avec au moins `50` exemples synthétiques par nœud `L2`.

### Méthode

La génération a été renforcée pour s'appuyer sur le contexte réel de chaque catégorie :

1. récupération des descendants réels de chaque couple `(L1, L2)` dans la taxonomie quand ils existent
2. utilisation de ces descendants comme ancres lexicales concrètes
3. repli sur les descriptions sémantiques `L2` quand le nœud `L2` est une feuille ou n'a pas de descendants
4. génération de plusieurs patrons de titres et descriptions par catégorie
5. conservation du style de surface du vrai `L1` :
   - marques observées
   - gamme de prix tronquée pour éviter les extrêmes
   - longueur moyenne des titres
   - longueur moyenne des descriptions

L'idée est d'obtenir un dataset plus exploitable pour :

- le contrôle `real vs synthetic`
- la séparation entre catégories sœurs
- un futur fine-tuning ou classifieur local `L2`

### Changements réalisés

- enrichissement du module `src/synthetic_generation.py`
- ajout du chargement des descendants de taxonomie `L2`
- amélioration des templates de génération
- limitation des prix extrêmes via une tranche centrée sur les quantiles du vrai `L1`
- génération d'un nouveau dataset synthétique à `50` lignes par `L2`
- génération d'un nouvel échantillon réel à annoter

### Fichiers concernés

- `src/synthetic_generation.py`
- `scripts/generate_synthetic_l2_dataset.py`
- `data/synthetic_l2_llm_targeted_50.csv`
- `data/real_l2_annotation_sample_llm_targeted.csv`
- `data/synthetic_l2_llm_targeted_50_summary.json`

### Résultats

- `191` catégories `L2` couvertes
- `9550` lignes synthétiques générées
- `393` lignes réelles exportées pour annotation
- nouveau champ de traçabilité :
  - `source = synthetic_llm_targeted`
  - `generation_method = targeted_l2_context_generation_v2`
  - `seed_descendants` pour garder les descendants taxonomiques ayant servi à la génération

### Limites

- la génération reste pilotée par templates, même si elle est maintenant bien plus ciblée
- certains nœuds très abstraits restent plus difficiles à rendre parfaitement naturels
- la qualité finale doit être mesurée par annotation réelle + validation downstream

### Prochaine étape

1. annoter une partie de `data/real_l2_annotation_sample_llm_targeted.csv`
2. lancer les 3 vérifications sur `data/synthetic_l2_llm_targeted_50.csv`
3. repérer les catégories `L2` où il faudra une génération encore plus spécifique ou un fine-tuning local

## 2026-04-28 - Outil d'annotation interactif L2

### Objectif

Faciliter l'annotation manuelle du vrai dataset `L2` avec un script simple à reprendre, sans interface lourde.

### Changements réalisés

- ajout d'un module d'annotation terminal
- ajout d'un script interactif qui sauvegarde après chaque action
- ajout d'un backup automatique quand le fichier est modifié en place
- possibilité de filtrer la session par `level_1_name`
- possibilité de reprendre sur les statuts `todo` et `review`

### Fonctionnalités du script

- affichage du produit courant :
  - hash
  - `level_1_name`
  - titre
  - description
  - marque
  - prix
  - candidats `L2`
- commandes disponibles :
  - numéro de candidat pour annoter
  - `s` pour passer
  - `r` pour marquer à revoir
  - `n` pour éditer une note
  - `c` pour vider l'annotation courante
  - `p` pour revenir à la ligne précédente
  - `q` pour sauvegarder et quitter

### Fichiers concernés

- `src/annotation_tool.py`
- `scripts/annotate_l2_dataset.py`
- `src/__init__.py`

### Résultats

- annotation possible directement en terminal
- sauvegarde immédiate après chaque ligne annotée
- ajout de la colonne `annotation_updated_at` si absente

### Limites

- outil terminal seulement
- pas encore d'interface web ou de vue multi-colonnes

### Prochaine étape

Utiliser ce script pour annoter un premier lot de données réelles puis lancer les trois validations du dataset synthétique.

## 2026-04-28 - Multi-prototypes hiérarchiques et beam search borné

### Objectif

Implémenter une méthode plus maligne que le fine-tuning supervisé synthétique direct pour `L2+` :

- plusieurs prototypes textuels par catégorie
- un seul embedding produit
- scoring local seulement sur les enfants valides du parent courant
- beam search borné pour ne descendre plus profondément que lorsque l'ambiguïté le justifie

### Principe retenu

Pour chaque catégorie, on fabrique plusieurs textes de référence :

- nom de catégorie
- chemin complet de taxonomie
- contexte parent
- description enrichie
- résumé des descendants
- variantes lexicales
- exemples synthétiques courts quand ils existent

On encode ensuite ces prototypes une fois pour toutes et on stocke leurs vecteurs.

À l'inférence :

1. on encode le produit une seule fois
2. on récupère uniquement les prototypes des enfants valides du parent courant
3. on calcule les similarités cosinus via produit matriciel sur des vecteurs normalisés
4. on agrège par catégorie avec `max` ou `mean_top_k`
5. si les deux meilleurs `L2` sont trop proches, on ouvre un beam de taille `2`
6. on regarde les enfants `L3`, puis éventuellement `L4`, mais pas au-delà de la profondeur choisie

### Changements réalisés

- ajout d'un module principal pour la construction des prototypes et l'inférence hiérarchique
- ajout d'un script de build de l'index de prototypes
- ajout d'un script de prédiction hiérarchique
- ajout d'un mode `--texts-only` pour inspecter les prototypes sans lancer d'encodage
- correction de la logique du beam search pour forcer au moins une expansion enfant en cas d'ambiguïté `L2`

### Fichiers concernés

- `src/hierarchical_prototype_search.py`
- `scripts/build_hierarchical_prototype_index.py`
- `scripts/predict_with_hierarchical_prototypes.py`
- `src/__init__.py`

### Résultats

- construction des textes de prototypes testée avec succès
- `71876` lignes de prototypes textuels générées dans un test de build
- scoring local conçu pour éviter de comparer un produit à toute la taxonomie globale
- beam search borné implémenté et validé sur un index jouet

### Limites

- la qualité finale dépendra encore du choix du modèle d'embeddings
- les prototypes des niveaux profonds restent plus pauvres quand la taxonomie ne fournit pas beaucoup de contexte
- le vrai benchmark métier nécessite un lot annoté `L2`

### Prochaine étape

1. construire l'index complet avec le modèle d'embeddings choisi
2. lancer des prédictions sur un lot de produits avec `level_1_name` déjà connu
3. comparer cette approche à une baseline barycentre simple

## Convention pour la suite

À partir de maintenant, chaque nouvelle étape importante du projet doit mettre à jour ce fichier.

## 2026-04-29 - Migration de l'enrichissement de catégories vers un LLM local Hugging Face

### Objectif

Remplacer l'enrichissement de catégories via API OpenAI-compatible par une génération locale avec un modèle Hugging Face, afin de rendre le pipeline autonome localement et d'exploiter `mps` sur Apple Silicon quand c'est disponible.

### Changements réalisés

- remplacement du provider `local` heuristique par un vrai backend LLM local basé sur `transformers`
- intégration du modèle par défaut `Qwen/Qwen2.5-7B-Instruct`
- génération via `AutoTokenizer` + `AutoModelForCausalLM`
- formatage du prompt en mode chat avec `apply_chat_template` quand le tokenizer le supporte
- sélection automatique du device local :
  - `mps` en priorité
  - sinon `cuda`
  - sinon `cpu`
- activation de `PYTORCH_ENABLE_MPS_FALLBACK=1` pour éviter qu'une opération non supportée par `mps` casse tout le run
- ajout d'options CLI pour piloter :
  - le modèle
  - le device
  - le `torch_dtype`
- conservation du provider `github-models` pour compatibilité, mais l'usage par défaut du script pointe maintenant vers le LLM local

### Fichiers concernés

- `src/category_enrichment.py`
- `scripts/generate_category_enrichment.py`
- `rapport/journal_projet.md`

### Résultats

- le pipeline d'enrichissement n'est plus dépendant par défaut d'une API distante
- la génération locale est prête pour `Qwen/Qwen2.5-7B-Instruct`
- le code est préparé pour utiliser `mps` automatiquement sur Mac compatible
- le format de sortie CSV reste inchangé pour ne pas casser la suite du pipeline

### Limites

- le modèle n'a pas été téléchargé ni exécuté end-to-end dans cette session car l'environnement courant n'a pas d'accès réseau ouvert pour récupérer les poids Hugging Face
- sur la machine courante, `torch.backends.mps.is_available()` remonte `False`, donc je n'ai pas pu valider un run réel sur `mps`
- un modèle `7B` sur machine locale demande suffisamment de RAM / mémoire unifiée pour être confortable

### Prochaine étape

1. télécharger localement `Qwen/Qwen2.5-7B-Instruct`
2. lancer une génération courte sur quelques catégories avec `--limit-per-level`
3. vérifier le device effectivement utilisé dans les logs
4. comparer qualitativement les sorties locales avec les anciennes sorties API

## 2026-04-29 - Migration du backend local vers MLX pour éviter les limites mémoire MPS

### Objectif

Remplacer le backend local `torch + transformers` par un backend `MLX` mieux adapté à Apple Silicon, après apparition d'une erreur de mémoire côté PyTorch `MPS` pendant la génération locale.

Erreur observée :

```text
RuntimeError: MPS backend out of memory (MPS allocated: 9.06 GiB, other allocations: 384.00 KiB, max allowed: 9.07 GiB). Tried to allocate 129.50 MiB on private pool.
```

### Pourquoi MLX

`MLX` est la stack machine learning d'Apple pensée pour Apple Silicon.

Dans ce projet, elle est utile pour plusieurs raisons :

- meilleure exploitation de la mémoire unifiée Mac
- backend natif Metal plus cohérent que le chemin `PyTorch MPS` dans ce cas d'usage
- support naturel des modèles quantifiés `4-bit`
- réduction du risque d'erreur de type `MPS backend out of memory`

L'idée n'est pas seulement d'accélérer, mais surtout de rendre la génération locale plus stable sur une machine à mémoire limitée.

### Changements réalisés

- abandon du chargement local `transformers + torch`
- passage du provider `local` à un chargement `MLX` paresseux
- ajout d'un téléchargement explicite du modèle dans un dossier local réutilisable
- chargement depuis un chemin local si le modèle est déjà présent, pour éviter tout re-téléchargement
- ajout d'un mode `--download-only` pour précharger le modèle avant génération
- ajout d'une détection `MLX LM` vs `MLX VLM` selon le modèle local
- ciblage du modèle :
  - `mlx-community/Qwen3.5-9B-MLX-4bit`

### Fichiers concernés

- `src/category_enrichment.py`
- `scripts/generate_category_enrichment.py`
- `rapport/journal_projet.md`

### Résultats attendus

- meilleure tenue mémoire sur Mac Apple Silicon
- réutilisation du modèle déjà téléchargé
- plus de dépendance au backend `torch` pour l'enrichissement local
- pipeline plus robuste pour des modèles quantifiés MLX

### Limites

- le modèle choisi `mlx-community/Qwen3.5-9B-MLX-4bit` est publié comme modèle `MLX VLM`, donc il faut garder en tête que l'intégration dépend du support `mlx-vlm`
- dans la session sandbox actuelle, MLX ne peut pas être exécuté réellement car l'accès Metal n'est pas exposé
- le téléchargement du modèle reste nécessaire une première fois car les poids font plusieurs gigaoctets

### Prochaine étape

1. installer la dépendance runtime exacte côté MLX si nécessaire
2. télécharger le snapshot local du modèle
3. lancer un test court avec `--limit-per-level`
4. vérifier la stabilité mémoire sur le Mac cible

## 2026-04-29 - Correction du choix de modèle MLX pour un pipeline texte-only

### Objectif

Corriger l'échec de chargement observé avec `mlx-community/Qwen3.5-9B-MLX-4bit` dans le pipeline d'enrichissement texte.

### Diagnostic

Le problème n'était pas que le code "n'utilisait pas MLX".

Le backend utilisait bien la branche MLX locale, mais le modèle ciblé passait par `mlx_vlm`, donc par un processor vision/vidéo `transformers` de type `Qwen3VL...`, ce qui introduisait des dépendances inutiles pour un usage purement texte et provoquait l'erreur sur `torchvision`.

### Changement réalisé

- changement du modèle MLX par défaut vers un vrai modèle texte `mlx-lm`
- nouveau défaut :
  - `mlx-community/Qwen2.5-7B-Instruct-4bit`
- correction de l'heuristique de détection `lm` / `vlm` pour ne plus classer abusivement les modèles `qwen3.5` comme modèles vision si la config ne contient pas de `vision_config`

### Résultat attendu

- chargement via `mlx_lm` au lieu de `mlx_vlm`
- plus de dépendance à `torchvision` pour l'enrichissement de catégories
- pipeline cohérent avec un besoin strictement texte

## 2026-04-30 - Accélération du backend local MLX

### Objectif

Améliorer le débit de génération du backend local MLX sans recharger les poids du modèle à chaque catégorie.

### Clarification importante

Les poids du modèle ne sont pas rechargés à chaque catégorie dans le pipeline actuel.

Ils sont chargés une seule fois au démarrage du script, lors de l'initialisation du client local, puis réutilisés pendant toute la boucle de génération.

Le temps principal venait donc surtout :

- du nombre de catégories à traiter
- de la longueur des prompts
- du nombre de tokens générés
- du fait que le backend local traitait encore les catégories une par une

### Changements réalisés

- ajout d'un vrai chemin batch pour le backend `mlx_lm` via `batch_generate`
- tokenisation explicite des prompts chat avant génération pour aligner le mode single et batch
- réduction des defaults du script pour accélérer un run standard :
  - `max_children = 6`
  - `max_descendants = 6`
  - `request_max_tokens = 240`
  - `request_batch_size = 4`
- ajout d'un rappel dans le CLI que le backend local MLX supporte maintenant la vraie génération batch

### Fichiers concernés

- `src/category_enrichment.py`
- `scripts/generate_category_enrichment.py`
- `rapport/journal_projet.md`

### Résultat attendu

- meilleure occupation du backend MLX
- moins d'overhead Python par catégorie
- prompts plus courts donc préfill plus rapide
- moins de tokens à générer donc latence plus basse

## 2026-04-30 - Simplification des prototypes pour accélérer encore la génération

### Objectif

Réduire encore le temps par catégorie en supprimant les champs jugés redondants dans les prototypes et dans la sortie LLM.

### Décision

Les champs suivants sont considérés comme peu utiles ou trop redondants dans ce pipeline :

- `children_summary`
- `descendants_summary`
- `enriched_description`

Le signal conservé dans les prototypes est maintenant centré sur :

- le nom de catégorie
- une liste courte de descendants taxonomiques
- les `lexical_variants`

### Changements réalisés

- le LLM ne génère plus que `lexical_variants`
- les champs de résumé et de description sont laissés vides
- les textes de référence gardent principalement :
  - `category_name`
  - `descendant_names_text`
  - `lexical_expansion`
- réduction des defaults :
  - `max_children = 5`
  - `max_descendants = 5`
  - `min_lexical_variants = 5`
  - `request_max_tokens = 120`

### Résultat attendu

- prompts plus courts
- sorties plus courtes
- moins de redite dans les prototypes
- réduction nette du temps moyen par catégorie

## 2026-04-30 - Suppression du JSON libre pour le backend local MLX

### Objectif

Réduire encore les erreurs de format du modèle local en évitant de lui demander du JSON libre.

### Changement réalisé

- le backend local MLX ne demande plus un objet JSON
- il demande maintenant uniquement une liste de variantes lexicales :
  - une ligne par variante
  - sans numérotation
  - sans puces
  - sans commentaire
- le code reconstruit ensuite `lexical_variants` localement à partir des lignes générées

### Résultat attendu

- moins d'erreurs de parsing
- moins de tokens gaspillés dans de la structure JSON
- meilleure robustesse en génération batch

## 2026-04-30 - Script Kaggle unifié pour fine-tuning GPU avec catégories enrichies

### Objectif

Préparer un script unique à exécuter dans un kernel Kaggle pour fine-tuner des embeddings sur GPU, en intégrant directement les catégories enrichies dans le pipeline d'entraînement et de validation.

### Changements réalisés

- création du dossier `kaggle/`
- ajout d'un script autonome :
  - `kaggle/train_finetune_with_enriched_categories.py`
- le script prend un dataset déjà préprocessé en entrée
- les hyperparamètres importants sont modifiables par CLI :
  - modèle d'embeddings
  - batch size
  - epochs
  - learning rate
  - weight decay
  - warmup
  - longueur max
  - colonnes de chemin taxonomique
  - types de prototypes catégories utilisés en train et en validation
- les catégories enrichies sont injectées dans le train comme exemples additionnels de la même classe
- sauvegarde d'un dossier d'expérience dont le nom encode les hyperparamètres principaux
- enregistrement :
  - `config.json`
  - `train_step_history.csv`
  - `epoch_metrics.csv`
  - checkpoints à chaque epoch
  - `best_model/`
  - `final_model/`
  - `summary.json`
  - prédictions de validation par retrieval

### Validation ajoutée

La validation finale compare les embeddings produits à des prototypes de catégories enrichies construits à partir de :

- `category_name`
- `path_text`
- `parent_context`
- `enriched_description`
- `descendant_names_text`
- `lexical_expansion`

Le script produit aussi une baseline `category_name` seule pour comparer facilement l'apport des enrichissements.

### Résultat attendu

- entraînements reproductibles sur GPU Kaggle
- comparaison facile entre plusieurs modèles d'embeddings et jeux d'hyperparamètres
- meilleure cohérence entre fine-tuning produit et espaces de prototypes catégories




il faudrait comparait plusieur models d'embeddings pour levels 2, regarder leur niveau de performance , 
aussi regarder si task pour le model d'embeddings change comment on arrive à le fine tunner

ensuite regarder aussi les resultats avec et sans beam search et combien de temps le beam search ajoute

regarder si l'enrichised catégories permet d'avoir un meilleur precisions sur la prediction level 1