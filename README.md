# Locally run open-weight LLMs for smart-city MLOps pipelines

[![DOI](https://zenodo.org/badge/1453741027.svg)](https://doi.org/10.5281/zenodo.24131237)

Code and data for **"Sovereign LLM-Augmented MLOps for Smart Cities: Can Locally Run Open-Weight Models
Generate Executable CI/CD Pipelines for Air-Quality Monitoring?"** (I. Essadeq, S. Nouh, K. Kandali),
submitted to ICISTA 2027.

Four open-weight models, served locally by [Ollama](https://ollama.com) on a laptop without a GPU
(Intel Core i7-8665U, 16 GB RAM), generate the same MLOps artifact: a Python module `pipeline.py`
(load, preprocess, train a random-forest anomaly detector with MLflow), a pytest file `test_pipeline.py`
and a GitHub Actions workflow, for the UCI Air Quality data. The generated tests are **executed** in a
pinned environment, and static detectors check reproducibility (RVH) and infrastructure (IIH) defects.

| Model (Ollama tag, Q4_K_M) | All tests pass | RVH | Median s / generation |
|---|---|---|---|
| GPT-4o (API, reference) | 3/10 | 0/10 | – |
| `qwen2.5-coder:3b` | 1/20 | 4/20 | 165 |
| `qwen2.5-coder:7b` | 1/20 | 9/20 | 252 |
| `llama3.1:8b` | **14/20** | 0/20 | 271 |
| `mistral:7b` | 0/20 | 17/20 | 211 |

## Layout

```
scripts/icista_m2_benchmark.py   generation (Ollama), extraction, pytest execution, RVH/IIH detectors
scripts/detectors.py             RVH / IIH detectors (copied verbatim from our MMC study)
scripts/analyze_icista.py        pass rates, 95% Wilson intervals, Fisher tests -> statistics_icista.json
scripts/classify_failures.py     failure taxonomy from the pytest outputs -> failure_categories.json
scripts/make_figure_icista.py    outcome figure (failure categories per model)
scripts/make_figures_extra.py    protocol diagram and cost-quality figure
results/runs_<model>/            one folder per model: results.json (settings, hashes, Ollama model details,
                                 per-run summary) and run_XX/ with raw_llm_output.txt, pipeline.py,
                                 test_pipeline.py and analysis.json (pytest output, detector verdicts, cost)
results/runs_gpt4o_replay/       GPT-4o reference: results and pytest outputs only (see below)
results/air_quality.csv          cleaned and labelled UCI Air Quality data used by every run
figures/fig1.pdf, fig1.png       Fig. 1
```

## Reproducing the paper's numbers (no model call)

```bash
pip install -r requirements.txt
cd results
python ../scripts/analyze_icista.py        # Table I
python ../scripts/classify_failures.py    # failure categories
python ../scripts/make_figure_icista.py   # -> ../figures/fig1.pdf
```

## Generating new artifacts (local, free)

1. Install Ollama and pull the models, e.g. `ollama pull llama3.1:8b`.
2. Prepare the pinned execution environment: Python 3.7.7, pytest 7.4.4, MLflow 1.30.1,
   scikit-learn 1.0.2, pandas 1.3.5, numpy 1.21.6.
3. Download the raw UCI Air Quality file (`AirQualityUCI.csv`) and run, from `results/`:

```bash
python ../scripts/icista_m2_benchmark.py --backend ollama --model llama3.1:8b --n 20 --out runs_new \
       --raw-csv <path/to/AirQualityUCI.csv> --pytest-python <path/to/python3.7>
```

The default values of `--raw-csv` and `--pytest-python` in the script are the paths of the original machine;
the script is kept byte-identical to the version whose SHA-256 is recorded in every `results.json`.
On a CPU-only laptop, close memory-hungry applications (a browser with many tabs) and disable sleep:
a 7B-8B model needs about 5 GB of free RAM to load.

## GPT-4o reference

The ten GPT-4o generations come from an earlier study of ours that is under double-blind review. Their
pytest outputs, detector verdicts and pass/fail results are included here, so that Table I and Fig. 1 can be
recomputed; the generated code itself will be added once that review is complete.

## Data

UCI Air Quality dataset (S. De Vito et al., Sensors and Actuators B 129(2), 2008,
doi:10.1016/j.snb.2007.09.060), distributed by the UCI Machine Learning Repository under CC BY 4.0.
`results/air_quality.csv` is the cleaned and labelled version (7,258 records; anomaly = CO > 4.0 or NO2 > 200).

## Citation and license

Archived on Zenodo: [10.5281/zenodo.24131237](https://doi.org/10.5281/zenodo.24131237) (this DOI always resolves to the latest version). See `CITATION.cff`. Code: MIT (see `LICENSE`). Model outputs in `results/` are released for research
reproducibility; each model is subject to its own license.
