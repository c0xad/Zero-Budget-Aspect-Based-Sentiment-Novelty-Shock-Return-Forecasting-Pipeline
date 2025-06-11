# ABSSENT – Progress Tracker

> **Legend**: ✓ = Done ▢ = Pending / In-Progress

---

## ✅ Completed So Far

| Area | Task | Status |
|------|------|--------|
| **Repository Bootstrap** | Create full directory tree & Python package skeleton | ✓ |
| | Generate MIT License, README, requirements.txt, pyproject.toml & setup.py | ✓ |
| **Utilities** | `config.py`, `logging.py`, `io.py` with rich logging & I/O helpers | ✓ |
| **ETL** | `etl/__init__.py` aggregation hub | ✓ |
| | EDGAR 8-K async scraper (`edgar.py`) with rate-limit + caching | ✓ |
| | Price loader (`prices.py`) with yfinance bulk download & indicators | ✓ |
| | News collector (`news.py`) with RSS feeds & ticker tagging | ✓ |
| **NLP** | Package init (`nlp/__init__.py`) | ✓ |
| | FinBERT sentiment analyzer (`sentiment.py`) w/ batching & caching | ✓ |
| | Aspect extractor (`aspects.py`) – keyword & embedding based classification | ✓ |
| | Novelty detector (`novelty.py`) – sentence-embedding novelty score & rolling window | ✓ |
| **CLI** | Typer-based CLI (`cli.py`) – download, analyze, predict, serve, demo | ✓ |
| **Examples** | End-to-end demonstration script (`example.py`) | ✓ |
| **House-Keeping** | Added `LICENSE` (MIT) and empty data folders | ✓ |

---

## 🔜 Next Steps (Week-by-Week Roadmap)

1. **Feature Engineering**
   ▢ Build `features/engineering.py` to merge sentiment, novelty & market factors.  
   ▢ Persist feature matrices to `features/` Parquet & DuckDB.

2. **Modeling Layer**
   ▢ Create `modeling/forecasting.py` with:
      • Panel regression (linearmodels)  
      • Walk-forward split & sklearn pipelines  
      • Backtesting helpers (vectorbt)

3. **API & Serving**
   ▢ FastAPI app `api.py` with `/predict`, `/health` + Pydantic schemas.  
   ▢ Dockerfile & optional Streamlit dashboard.

4. **Workflow Orchestration**
   ▢ Prefect flows (`flows/`) to chain ETL → NLP → Features → Model → Report.  
   ▢ GitHub Actions scheduled run.

5. **Testing & Quality**
   ▢ Add pytest unit & integration tests for every module.  
   ▢ Configure Ruff, Black, MyPy in CI workflow.

6. **Documentation**
   ▢ MkDocs site with material theme, API docs via `mkdocstrings`.  
   ▢ Tutorial notebooks under `docs/examples/` executed in CI.

7. **CI / CD**
   ▢ GitHub Actions matrix testing, coverage badge (≥90 %).  
   ▢ PyPI publish & Docker image push on tagged release.

8. **Data Governance & Compliance**
   ▢ Add `docs/compliance/sec_rate_limit.md` & automate User-Agent injection tests.

9. **Community**
   ▢ CODEOWNERS, contributor covenant, issue templates, roadmap board.

---

*Last updated: 2025-06-11*
