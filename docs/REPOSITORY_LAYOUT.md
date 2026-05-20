# Repository Layout

Recommended repository structure for a clean public GitHub version of the project.

```text
Criteo/
├── README.md
├── pyproject.toml
├── taxonomy.txt
├── scripts/
│   ├── *.py
│   ├── lib/
│   └── kaggle/
├── docs/
│   ├── PROJECT_HANDOVER.md
│   └── REPOSITORY_LAYOUT.md
├── latex/
│   ├── rapport_final/
│   └── note_synthese/
├── legacy/
│   ├── scripts/
│   ├── kaggle/
│   ├── src_archive/
│   └── *.ipynb
├── resarch_paper/
├── rapport/
├── dataset/      # git-ignored
├── data/         # git-ignored
├── models/       # git-ignored
├── artifacts/    # git-ignored
└── checkpoints/  # git-ignored
```

## What Should Be Versioned

- `README.md`
- `pyproject.toml`
- `scripts/`
- `docs/`
- `latex/`
- `legacy/`
- `taxonomy.txt`
- optionally `rapport/` and `resarch_paper/` if they are useful as academic context

## What Should Stay Out of Git

- `data/`
- `dataset/`
- `models/`
- `artifacts/`
- `checkpoints/`
- Python caches, virtual environments, and temporary LaTeX outputs

## Installation with uv

Base environment:

```bash
uv sync
```

With local MLX support:

```bash
uv sync --extra local-mlx
```

With development extras:

```bash
uv sync --extra dev
```

To generate a lockfile:

```bash
UV_CACHE_DIR=.uv-cache uv lock
```
