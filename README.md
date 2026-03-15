# 🛒 Product Categorization at Scale : From Unsupervised to Supervised

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Compatible-ee4c2c.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Supervised-green.svg)

**Projet Stat'App – ENSAE Paris (2025-2026) en collaboration avec Criteo.**

Ce projet vise à projeter automatiquement un catalogue massif de produits (plus de 120 000 références) vers la **Google Product Taxonomy**. Le dépôt documente notre transition d'une approche d'optimisation sous contrainte (non-supervisée) vers une architecture de pointe en Deep Learning à double modèle (Dual Embedding Pipeline).

## 👨‍💻 Équipe
* **Salah Eddine NEJJARI**
* **Tao DANG TRAN**
* **Thomas FAVRE**
* **Yohan ANDRIAMAMPIONONA**

**Encadrement :** Guillaume Lochon, Ilan Benchetrit (Criteo) | Nicolas Chopin (ENSAE)

---

## 🧠 Architecture du Projet

Le projet s'est déroulé en deux phases méthodologiques distinctes :

### Phase 1 : Assignation Globale par Flux Réseau (Sans Ground Truth)
Lors de la phase exploratoire, nous devions catégoriser les produits sans données annotées, en respectant uniquement des quotas volumétriques connus.
* **Vectorisation :** Modèle `all-MiniLM-L6-v2` (SentenceTransformers) pour encoder les produits et les barycentres des catégories de la taxonomie.
* **Optimisation :** Modélisation sous forme de graphe de transport biparti et résolution via l'algorithme **Min-Cost Max-Flow** (librairie `OR-Tools`) pour assigner les produits tout en respectant strictement les quotas.

### Phase 2 : Dual Embedding Pipeline (Avec Ground Truth)
L'obtention de données annotées nous a permis de développer un pipeline hiérarchique en deux étapes (*Two-Stage*) :
1. **Verrouillage de la Racine (Level 1) - Supervisé :** * Fine-Tuning du modèle d'embedding `mini-gte` via une **Batch Hard Triplet Loss**.
   * Classification supervisée par **XGBoost** et Régression Logistique sur les vecteurs résultants (atteignant **~91-94% d'Accuracy**).
2. **Raffinement Sémantique (Level 2+) - Zero-Shot :**
   * Modélisation de la taxonomie sous forme de graphe orienté (`NetworkX`).
   * Enrichissement sémantique des ancres Level 2 généré par LLM.
   * Retrieval sémantique strict (*Semantic Textual Similarity*) sur les enfants valides via le modèle `BAAI/bge-small-en-v1.5`.
   * Intégration d'un score de confiance (distance cosinus) permettant la mise en place d'un seuil de rejet (**Active Learning**).

---

## 🛠️ Technologies Utilisées

* **Manipulation & Graphes :** `pandas`, `numpy`, `networkx`
* **Machine Learning :** `scikit-learn`, `xgboost`
* **Deep Learning & NLP :** `torch`, `sentence-transformers`, `peft` (Parameter-Efficient Fine-Tuning)
* **Recherche Opérationnelle :** `ortools` (Google OR-Tools)

---

## 🚀 Installation et Utilisation

### 1. Cloner le dépôt
```bash
git clone [https://github.com/VOTRE_NOM_UTILISATEUR/criteo.git](https://github.com/VOTRE_NOM_UTILISATEUR/criteo.git)
cd criteo
```

### 2. Créer un environnement virtuel
```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Exécuter les notebooks
* **Phase 1 :** LV1.ipynb (Optimisation sous contrainte)
* **Phase 2 :** LV2.ipynb (Dual Embedding Pipeline)

### 📂 Structure du Répertoire

```
├── data/                   # Données brutes et générées (non pushées sur GitHub si trop lourdes)
├── saved_models/           # Modèles XGBoost et LabelEncoders sauvegardés (.joblib)
├── LV1.ipynb               # Notebook de la Phase 1 (OR-Tools)
├── README.md               # Ce fichier
└── requirements.txt        # Liste des dépendances Python
```

#### 🔮 Perspectives (Next Steps)

-   Implémentation d'un algorithme de Beam Search pour prévenir la propagation d'erreurs en cascade lors de la descente vers les Levels 3 et 4.

-   Évaluation comparative de la distance Euclidienne (L2) face à la similarité cosinus dans l'espace latent post-finetuning.

