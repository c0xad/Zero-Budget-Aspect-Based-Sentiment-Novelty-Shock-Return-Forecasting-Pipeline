# Zero-Budget Aspect-Based Sentiment & Novelty-Shock Return Forecasting Pipeline

## 0. Executive Summary
Build an end-to-end, fully reproducible research & production pipeline that forecasts next-day equity returns with **Aspect-Based Sentiment Shock** extracted from SEC 8-K filings (plus optional news). All components use **free** data sources and **open-source** libraries, run on commodity CPU or Google Colab.  Target deliverables:

* Peer-review-ready empirical paper (JFE style) & Jupyter reproducibility appendix.
* Python package (`abssent`) published to PyPI.
* Back-testable trading signals (CSV + REST endpoint).
* Public GitHub repo with CI/CD, code-cov, Docker image, and MIT license.

## 1. High-Level Architecture
```
raw-data/           <- unprocessed filings, headlines, price CSVs
└── edgar/
└── news/
processed/          <- parquet/feather intermediates after ETL
features/           <- daily firm-level feature matrices
models/             <- persisted sklearn/pytorch artifacts
reports/            <- Jupyter notebooks & HTML summaries
abssent/            <- pip-installable source code package
    ├── etl/
    ├── nlp/
    ├── features/
    ├── modeling/
    └── utils/
```  
Data & tasks orchestrated by **DAG** via `prefect` (zero-cost cloud) or `dagster-local`. Each run is identified by a Git SHA + UTC timestamp.

## 2. Technology Stack
1. Python ≥3.10 (type-hints, pattern-matching).
2. Poetry for dependency reproducibility.
3. PyTorch + HuggingFace 🤗 Transformers + Sentence-Transformers.
4. Pandas, Polars, NumPy, StatsModels, SciPy.
5. Prefect (or Dagster) for workflow orchestration.
6. DuckDB/Parquet for columnar storage (lightweight, SQL-friendly).
7. GitHub Actions for CI-lint-test-build-publish.
8. Docker for env parity; pre-build CPU & CUDA images.

## 3. Detailed Work Breakdown
### 3.1 Repository Bootstrap (Day 1)
```bash
# create repo, initialize poetry & pre-commit
poetry new abssent && cd abssent
poetry add pandas polars numpy scikit-learn statsmodels ...
pre-commit install
```
Todo:
* Add `.editorconfig`, `pyproject.toml`, `ruff`/`black` configs.
* Set up GitHub repo, default branch `main`, branch protection rules.
* Enable GitHub Actions template (pytest + coverage + docs build).
* Add `docs/` with MkDocs skeleton.

### 3.2 Data Ingestion Modules (Week 1-2)
#### 3.2.1 EDGAR 8-K Scraper
* Use SEC **Daily Index** text files. Endpoint format:  
  `https://www.sec.gov/Archives/edgar/daily-index/<YEAR>/QTR<q>/company.<YYYYMMDD>.idx`
* Parse with `pandas.read_fwf`, filter `form_type == "8-K"`.
* Download filing HTML/TXT via `https://www.sec.gov/Archives/` + `filename`.
* **Respect SEC rate-limit** (10 req/s) with `asyncio` + `aiohttp` + exponential back-off.
* Store raw files under `raw-data/edgar/<CIK>/<ACCESSION>.txt`.

#### 3.2.2 News Headline Collector (optional)
* Sources: Yahoo Finance RSS, FinViz screener news JSON, Google News API (rate-limited but free with API key).
* For each ticker, collect last N days; deduplicate by URL hash.

#### 3.2.3 Price Downloader
* `yfinance` bulk download per ticker list (S&P 1500) for last 10 years.
* Save AS-OF copy in `raw-data/prices/` (CSV) and mirrored DuckDB table.

Workflow: Prefect **ETL** flow runs nightly via GitHub Actions + `schedule: cron("0 6 * * *")` (6 AM UTC after EDGAR update).

### 3.3 Text Pre-Processing (Week 2-3)
* Convert HTML→plain text with `beautifulsoup4` & `boilerpy3`.
* Lower-case, remove tables & exhibits via regex heuristics.
* Split into paragraphs (`\n{2,}`) → list of strings.
* Detect language with `langdetect`, drop non-English.
* Cache cleaned paragraphs as `processed/edgar_clean/<ACCESSION>.parquet`.

### 3.4 Aspect Label Expansion
1. Seed dictionary (YAML file):
   ```yaml
   liquidity:
     - liquid*
     - cash flow
   guidance:
     - guidance
     - forecast
   management_change:
     - resignation
     - appoint*
   risk:
     - risk factor*
   ```
2. Load pre-trained GloVe (6B-100d).  For each seed token call `most_similar(topn=20)`; filter cosine > 0.65; write extended list to `data/lexicons/aspect_terms.json`.
3. Unit tests ensure no overlap between aspects > 5 %.

### 3.5 Aspect Assignment (Week 3-4)
* For each paragraph compute trigram TF-IDF vector (`scikit-learn`).
* Keyword match: if paragraph includes ≥1 aspect term add to candidate aspect list.
* Embedding fallback: use `SentenceTransformer("all-mpnet-base-v2")`; cosine similarity to mean embedding of seed terms per aspect; assign if cos > 0.4.
* Paragraph can map to multiple aspects.
* Output schema (`processed/aspect_map.parquet`):  
  `accession`, `paragraph_id`, `aspect`, `text`.

### 3.6 Sentiment Scoring (Week 4)
* Load `ProsusAI/finbert-sentiment` (3-way: pos/neg/neutral).
* Batch paragraphs by aspect into ≤64 token windows, pad/truncate.
* Use CPU inference (≈ 30 docs/s) or Colab GPU lane for re-processing.
* Store logits + label per paragraph.

### 3.7 Novelty Shock Computation (Week 4-5)
* For each firm-day, aggregate all paragraphs (any aspect) → list.
* Generate mean embedding via `sbert.encode` (pooling="mean").
* Maintain 30-day rolling window mean embedding (DuckDB window function).
* `novelty = 1 − cosine(today, rolling_mean)`.
* Persist firm-day novelty values in `features/novelty.parquet`.

### 3.8 Feature Engineering (Week 5)
For each firm-day:
```
AS_sent_{aspect} = mean(FinBERT_sentiment_logits · sentiment_weight)
AspectShock_{aspect} = AS_sent_{aspect} * novelty
```
* Optionally weight paragraphs by word-count.
* Merge with conventional factors: size, book-to-market (Fama-French via `wrds` clone or Ken French website), momentum.
* Output `features/factors.parquet` (wide format).

### 3.9 Modeling & Hypothesis Testing (Week 6-7)
#### 3.9.1 In-Sample Panel Regression
```python
import linearmodels.panel as plm
mod = plm.PanelOLS.from_formula(
    'ret_fwd ~ 1 + AspectShock_guidance + AspectShock_liquidity + MKT + SMB + HML + Momentum + EntityEffects + TimeEffects',
    data=df
)
res = mod.fit(cov_type='clustered', cluster_entity=True)
```
* Save summary to `reports/reg_tables/` (LaTeX + HTML).

#### 3.9.2 Granger Causality
* `statsmodels.tsa.stattools.grangercausalitytests` on firm equal-weighted index.
* Lag selection by BIC.

#### 3.9.3 Out-of-Sample Forecast & Backtest
* Expanding-window walk-forward split (70 % train, 30 % test).
* Evaluate MAE, MSE, Hit-Rate (>0 return), Sharpe of long-short top/bottom decile.
* `vectorbt` for backtesting.

### 3.10 Packaging & API (Week 8)
* Wrap inference into `abssent/predict.py` with CLI:
  ```bash
  abssent predict --date 2025-06-12 --tickers AAPL MSFT --out predictions.csv
  ```
* FastAPI micro-service (`app.py`) exposes `/predict` endpoint returning JSON.
* Dockerfile:
  ```dockerfile
  FROM python:3.10-slim
  COPY . /app
  RUN pip install .[all]
  CMD ["uvicorn","abssent.app:app","--host","0.0.0.0","--port","8080"]
  ```

### 3.11 Continuous Integration
* GitHub Actions matrix: {ubuntu-latest, windows-latest} × Python {3.10,3.11}.
* Jobs: `ruff check`, `black --check`, `pytest`, `poetry build`, `docker_build_push` (to GHCR).
* Coveralls badge ≥90 %.

### 3.12 Documentation
* MkDocs with `material` theme.
* Autodoc via `mkdocstrings[python]`.
* Tutorials Jupyter notebooks under `docs/examples/`, executed by `nbdev-myst` in CI.

### 3.13 Deployment (Optional)
* HuggingFace Spaces (free CPU) hosting the FastAPI app.
* Create a Streamlit dashboard for interactive exploration.

## 4. Milestone Timeline (Aggressive 8-Week Plan)
| Week | Deliverable |
|------|-------------|
| 1 | Repo bootstrapped, CI green, EDGAR & price ETL MVP |
| 2 | Complete ETL; cleaned paragraph dataset |
| 3 | Aspect lexicon + assignment pipeline |
| 4 | FinBERT sentiment inference + novelty scores |
| 5 | Feature table ready; exploratory data analysis report |
| 6 | Panel regression & robustness checks |
| 7 | Out-of-sample backtest, API prototype |
| 8 | Packaging, docs, Docker & optional dashboard |

## 5. Risk & Mitigation
* **SEC rate-limit bans** → implement politeness delay & user-agent header.
* **GPU quota on Colab** → fall back to CPU batch inference overnight.
* **Data revisions (EDGAR restatements)** → store accession-level hashes; append-only.
* **Look-ahead bias** → enforce trade-date alignment; use `prices.shift(-1)` for forward returns.

## 6. Future Extensions
* Incorporate earnings-call transcripts (Seeking Alpha API).
* Use instructor-tuned models (`instructor-xl`) for zero-shot aspect extraction.
* Reinforcement Learning trading agent via `stable-baselines3`.
* Compare vs. BERT-based volatility forecasting.

## 7. Zero-Cost Reproducibility & Experiment Tracking
### 7.1 Deterministic Environments
* **`poetry lock --no-update`** to freeze versions; commit lockfile.
* Provide Conda `.yml` export (`poetry export --format=yaml`) for Colab/Kaggle one-click recreate.
* Optional: include a `devcontainer.json` (VS Code Remote-Containers) so contributors spin up an identical environment with *Docker Desktop* Free tier.

### 7.2 MLflow Local Server
* Add `mlflow` (MIT) as an optional dependency.
* Launch with SQLite backend & local artifact store:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
  ```
* GitHub Action artifact upload of **`mlruns/`** directory on each merge so experiment history is preserved without paid hosting.

### 7.3 DVC (Data Version Control) – Local Remote
* Track large Parquet files with `dvc`; remote set to `./dvc-storage` (inside repo, Git-ignored).
* Contributors pull data via `dvc pull` – zero external cost.

### 7.4 Binder & Google Colab Badges
* Add `launch binder` & `Open in Colab` badges in `README.md` auto-generating an environment from `environment.yml` using Binder (mybinder.org, free) and Colab (requirements parsing).

## 8. Extended Statistical Tests (Zero-License-Fee Libraries)
1. **Fama-MacBeth Cross-Sectional Regression** – implement with `linearmodels` free version.
2. **Quantile Regression** – `statsmodels.regression.quantile_regression.QuantReg` for tail impact.
3. **Bootstrap Confidence Intervals** – `arch.bootstrap` (BSD-3) for firm-level clustered bootstrap.
4. **Newey-West HAC** standard errors – built-in to `statsmodels`.
5. **PurgedWalkForwardCV** to eliminate label leakage – DIY class in `abssent/modeling/cv.py`.

## 9. Memory & Compute Optimisation (Works on 8 GB RAM Laptop)
* Chunk EDGAR parsing with generators; never load full filing list into RAM.
* Use `polars.scan_parquet()` lazy queries to filter columns before collect.
* For BERT inference: gradient-disabled (`model.eval(); torch.no_grad()`) and batch-size tuned to available RAM.
* Cache sentence embeddings in DuckDB **`CREATE TABLE AS SELECT`**; subsequent runs skip costly encoding.

## 10. Data Governance, Ethics & Compliance
* Add **`docs/compliance/sec_rate_limit.md`** summarising 17 CFR 240.24b-2 fair-use; include polite User-Agent string `abssent-research-bot/1.0 (+https://github.com/you/abssent)`.
* Strip personal-identifiable information (PII) from filings (rare but possible in exhibits) with regex `SSN|taxpayer` alert and manual review script.
* Publish a **Code of Conduct** (Contributor Covenant v2.1) and **Data Usage Disclaimer** (not investment advice).

## 11. Community & Governance
* **GitHub Issue Templates**: _Bug_, _Feature Request_, _Research Idea_.
* **Conventional Commits** enforced via `commitlint` (Node 18 included in GitHub runner; zero cost).
* `CODEOWNERS` file routing PRs touching `abssent/nlp/` to *NLP maintainers* group.
* Monthly live coding session on **Twitch** or **YouTube Live** – free distribution.

## 12. Free Compute Playbook
| Platform | Quota | How We Use It |
|----------|-------|---------------|
| Google Colab | 12 h GPU sessions, free | Heavy FinBERT inference (`colab/inference.ipynb`). |
| Kaggle Notebooks | 30 GB RAM, 40 h/week | Backtesting notebook with `vectorbt`. |
| Microsoft Azure-for-Students | $100 credit/year (optional, still zero cash) | Burst large ETL via Spot VMs. |
| HPC via NSF XSEDE allocations (educational) | Apply for free CPU hours | Run cross-validation grid search. |

Automation: CI creates an **`.ipynb`** that mounts Google Drive, pulls small model weights (<1 GB) from HuggingFace CDN, and triggers nightly inference via Colab *scheduled notebooks*.

## 13. Publication & Open Science Workflow
* `paper/` directory with **LaTeX** template (Overleaf free plan sync via GitHub).  Figures auto-exported from Jupyter via `matplotlib.pyplot.savefig('../paper/figs/figure1.png')`.
* Release candidate of dataset on **Zenodo** (up to 50 GB per DOI free) using GitHub-Zenodo webhook for archival DOI.
* Pre-submit to SSRN – zero fee.

## 14. Test Suite Deep Dive
* **Unit**: pytest + fixtures for fake 8-K HTML; aim ≥200 tests.
* **Property-Based**: `hypothesis` tests for text-cleaning idempotence.
* **Integration**: spin up FastAPI with `uvicorn` on random port; call `/predict`.
* **End-to-End**: workflow test via `prefect` `FlowRunner` against 3 sample tickers.
* **Static Analysis**: `ruff`, `bandit` (security), `interrogate` (docstring coverage).

## 15. Concrete Next Actions (Day-by-Day, Week 1)
| Day | Task |
|-----|------|
| 1 | Fork template repo, run initial `poetry install`, push first green CI. |
| 2 | Implement SEC downloader skeleton with retry + unit tests. |
| 3 | Finish price loader & DuckDB schema; commit sample Parquet. |
| 4 | Draft text-cleaning pipeline; verify on 5 filings; write blog post in `docs/blog/2025-cleaning-pipeline.md`. |
| 5 | Aspect lexicon YAML + expansion notebook; open PR for community seed suggestions. |
| 6 | Merge news collector; add timer metric via `rich.progress`. |
| 7 | End-of-week demo: push Colab badge & record 10-min Loom walkthrough (free). |

---
_All new features above are achievable with open-source licenses or free-tier services—keeping the project strictly zero-budget while enhancing depth, reproducibility, and community engagement._

---
©2025 You. Licensed under MIT.  Happy building! 🚀
