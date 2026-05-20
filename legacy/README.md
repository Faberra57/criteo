# Legacy Archive

Ce dossier conserve les éléments historiques du projet qui ne font plus partie du pipeline final.

## Principe

`legacy/` n'est pas la source de vérité du dépôt.
La source de vérité du pipeline final est maintenant dans :
- `scripts/`
- `scripts/lib/`
- `scripts/kaggle/`

Les fichiers archivés ici sont gardés pour :
- retracer les expérimentations passées ;
- relire d'anciennes approches ;
- récupérer une idée ou une implémentation si besoin.

## Contenu

- `legacy/scripts/` : anciens scripts exploratoires ou intermédiaires.
- `legacy/kaggle/` : anciennes variantes Kaggle non retenues.
- `legacy/src_archive/` : ancien package `src/`, conservé uniquement comme archive technique.
- `legacy/*.ipynb` : notebooks historiques.

## Règle d'usage

Ne pas repartir de `legacy/` pour continuer le projet.
Si un développement doit reprendre, partir de :
- `README.md`
- `docs/PROJECT_HANDOVER.md`
- `scripts/`

## Pourquoi ces fichiers ont été archivés

Les éléments présents ici correspondent typiquement à l'un des cas suivants :
- version exploratoire remplacée par une version plus récente ;
- script intermédiaire non retenu dans le pipeline final ;
- ancienne architecture avant le passage à `scripts/lib/` ;
- notebook d'analyse ponctuelle.
