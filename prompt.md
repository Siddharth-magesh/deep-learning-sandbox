Prompt for author (write GH Pages content that reads human-written, technical, and original):

Goal
- Produce a polished, human-authored GitHub Pages site for a multi-repo deep-learning sandbox. The site must feel authored by the project maintainers (no generic AI phrasing), be technically accurate, and be a useful reference for researchers and engineers.

Audience
- ML researchers, ML engineers, data scientists, and advanced students who want to understand, run, reproduce, or extend the projects and checkpoints.

Tone & Style Requirements
- Voice: first-person plural (we) with occasional first-person singular author notes for experiments; friendly, confident, and precise.
- Avoid AI-generated clichés (e.g., "As an AI language model", "This project uses state-of-the-art...") — instead use concrete specifics.
- Use active voice and short paragraphs. Include human context: motivation, trade-offs, why design choices were made.
- Add short contributor notes or "Author's note" blurbs where appropriate (e.g., about tricky training steps).
- Include explicit provenance: commit hashes or tag names for important releases referenced on the page.

Site Structure (recommended pages)
- Home (overview + quick links)
  - One-paragraph elevator summary of the overall workspace and philosophy.
  - Highlighted projects (clip, vision-transformer, generative-pretrained-transformer-2, DenseNet, Siamese network).
  - Quick-start bullets: Install, Run, Demo, Reproduce.
  - Badges: Python version, CI status, license, downloads or model-artifact size.

- Getting Started
  - Setup (tested OS, Python version, venv/conda commands).
  - Install dependencies (exact commands and pyproject/tip).
  - Data access and storage: instructions for dataset preparation, expected directory layout, and how to link checkpoints (use Git LFS or external URLs).
  - Quick commands: train a tiny test job, run evaluation on a checkpoint.

- Projects (one subpage per major folder)
  - For each subproject (e.g., `contrastive-language-image-pretraining`, `vision-transformers`, `generative-pretrained-transformer-2`, `siamese-network`, `residual-network`):
    - Short description & motivation (2–4 sentences).
    - Main scripts and entry points (exact file names and purpose).
    - Reproducible example: minimal command to run training for 1 epoch on a toy dataset; include expected console output snippet and how to interrupt/resume.
    - Where checkpoints are saved and which checkpoint to use for demos (list exact filenames).
    - Key hyperparameters and hardware used (GPU type, batch size, lr schedule).
    - Link to in-repo docs (e.g., API_REFERENCE.md, TRAINING_GUIDE.md).

- API Reference
  - Auto-generated or hand-curated list of core modules and functions (e.g., `train.py`, `evaluate.py`, `data_loader.py`, `vision_transformer.py`, `text_transformer.py`, `clip.py`).
  - For each core function/class: one-line purpose + arguments + an example snippet showing usage.
  - If full auto-generation is used, add a short human-written introduction and a "Notes" section for inconsistencies or caveats.

- Tutorials & Notebooks / Examples
  - Step-by-step tutorial for common tasks (training from scratch, fine-tuning a checkpoint, inference).
  - One short tutorial that reproduces a headline result from the repo (even if on a small dataset).
  - Include code blocks with copy-paste commands and expected run time for given hardware.

- Results & Experiments
  - Tables of metrics for major checkpoints (accuracy, loss, validation protocol).
  - Short notes on experimental setup: dataset splits, augmentation, number of seeds.
  - If there are visualizations or sample outputs, include high-quality screenshots with captions and alt text.

- Reproducibility
  - Exact environment: Python + key package versions (pip freeze snippet or pyproject lock), GPU drivers tested.
  - Seeds and deterministic settings used.
  - How to reproduce major experiments end-to-end (commands + expected outputs + common errors and fixes).

- Checkpoints & Downloads
  - Inventory of checkpoints inside `checkpoints/` and guidance for usage and licensing.
  - For large models: point to external hosting or Git LFS instructions.
  - Note on model licensing and redistribution.

- Contributing
  - How to submit issues, PRs, and standards for commit messages.
  - Coding style, testing guidelines, and how to run local tests.
  - How to add new experiment logs and attach artifacts.

- License & Citation
  - Short license summary (link to LICENSE).
  - CITATION snippet: Provide BibTeX entries for models/papers used or authored by repo maintainers.
  - How to cite the repository in academic work.

- Appendix / Advanced Topics
  - Distributed training notes, optimizer choices, memory/perf tuning.
  - Troubleshooting (OOM, data loading bottlenecks).

Technical Content Requirements (what to include verbatim)
- Exact file names and script references (e.g., entrypoint commands such as `python src/train.py --config ...`), with runnable examples.
- Exact checkpoint filenames and where they live in the repo (or external URLs).
- CLI examples for reproducible runs: full commands, including env vars where required.
- Sample config YAML or JSON snippets for common workflows.
- Short code snippets (<= 30 lines) that demonstrate typical usage: loading a model, running inference, computing metrics.
- Minimal example outputs (logs, sample images, numeric metrics) so readers know what to expect.

Design & UX Guidance
- Prefer a clean developer-focused template: MkDocs Material or Jekyll with a technical theme.
- Top navigation: Home, Getting Started, Projects, Tutorials, API, Results, Contribute.
- Include a search (haunted by many pages).
- Make "Run a demo" prominent and easy.
- Use high-quality figures/screenshots; supply alt text and captions.
- Add keyboard-friendly code blocks with copy buttons.
- Add a changelog or release notes section linked to tags.

SEO, Metadata & Front Matter
- Provide page titles and concise meta descriptions for each page.
- For each tutorial or project page include YAML front-matter (title, description, authors, tags).
- Suggest canonical URLs and simple social preview images that include the repo name and one-line tagline.

Build & Deployment (concise steps to include)
- Preferred: Use GitHub Pages with a docs/ folder or `gh-pages` branch.
- Or recommend MkDocs + GitHub Actions deploy workflow: include the GH Action workflow example and exact `mkdocs build && mkdocs gh-deploy` commands.
- Include commands to build site locally and preview:
  - mkdocs serve
  - or jekyll serve
- Add instructions for updating site assets (images, large model files — note to host large binaries elsewhere).

Human Quality Checklist (for the author)
- Replace boilerplate phrases with concrete details and human commentary.
- Confirm all commands are runnable on a clean environment (provide any required secrets or tokens separately if needed).
- Add at least two short first-hand notes (e.g., “We observed X when training with Y…”).
- Verify links to in-repo docs (API_REFERENCE.md, TRAINING_GUIDE.md) resolve.
- Proofread for clarity; avoid passive filler and marketing talk.
- Run link-check and spell-check before publishing.

Deliverables (what you should produce)
- A ready-to-deploy set of markdown pages (one per section above), with images and code snippets embedded.
- A `README`-style homepage and a `CONTRIBUTING` summary for quick edits.
- A short `deploy.md` showing exact commands and a GitHub Actions snippet for automatic deploys.
- A final short editorial note confirming manual review and test steps completed.

Extra notes for the author
- If any dataset or checkpoint cannot be published, include a placeholder with instructions on how to request access.
- Add citations for any external models or datasets used.
- Keep content modular so maintainers can update individual project pages without reworking the whole site.

Acceptance criteria (how we know this is done)
- The site contains all pages listed above with real, repo-specific commands and checkpoint references.
- The writing reads like a human maintainer wrote it — includes reasoning, choices, and a few personal notes.
- Building locally succeeds with provided commands and shows the pages rendered correctly.

Use this prompt to produce the GH Pages markdown content, assets list, and a minimal deploy workflow.
