# -*- coding: utf-8 -*-
"""ICISTA 2027 -- open-weight LLMs on the executable M2 task (smart-city air quality).

Same task as the IJoSI study (pytest_m2_standalone.py): the model writes pipeline.py,
test_pipeline.py and a GitHub Actions workflow for a RandomForest anomaly detector on the UCI
Air Quality data; the generated tests are then executed with pytest in the pinned M2
environment (Python 3.7.7, pytest 7.4.4, MLflow 1.30.1). Static RVH/IIH detectors (MMC, copied
verbatim) are applied to the same code.

Backends
  --backend ollama   local open-weight model served by Ollama (http://localhost:11434), free
  --backend replay   re-executes saved generations (the 10 GPT-4o runs of the IJoSI study),
                     so every model is executed in the SAME environment on the SAME data
  --backend fake     offline dry run of the whole chain

Examples (from ICISTA/code):
  py -3.13 icista_m2_benchmark.py --backend fake --n 2 --out runs_dry
  py -3.13 icista_m2_benchmark.py --backend replay --replay-dir "<.../m2_generations/final_batch_...>" --out runs_gpt4o_replay
  py -3.13 icista_m2_benchmark.py --backend ollama --model qwen2.5-coder:7b --n 10 --out runs_qwen25coder7b
"""
from __future__ import annotations
import argparse, ast, datetime, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detectors import detect_rvh, detect_iih

# ---------------------------------------------------------------------------
# Fixed protocol (identical to the IJoSI M2 check)
# ---------------------------------------------------------------------------
TEMPERATURE, MAX_TOKENS = 0.3, 1500          # as in pytest_m2_standalone.py (GPT-4o)
SYSTEM = "You are an MLOps engineer."
FEAT = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
        'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']
PROMPT = ("You are an MLOps engineer for a smart city air quality system. Generate a production-ready CI/CD "
          "pipeline for a RandomForest anomaly detector. Features=%s, target column 'Anomaly', data at "
          "'data/air_quality.csv'. Provide: (1) a Python module pipeline.py with functions load_data(file_path), "
          "preprocess_data(data) -> (X,y), train_model(X_train,y_train); (2) a pytest file test_pipeline.py importing "
          "from pipeline; (3) a GitHub Actions YAML. Use MLflow and error handling." % FEAT)

THIRD_PARTY = {"pandas", "numpy", "sklearn", "scipy", "mlflow", "pytest", "joblib", "xgboost", "matplotlib", "yaml"}
PINNED_ENV = set(getattr(sys, "stdlib_module_names", set())) | THIRD_PARTY      # same set as MMC / ICBDAIA
DEFAULT_PYTEST_PY = r"C:\Program Files\Odoo 16.0.20230507\python\python.exe"   # the pinned M2 environment


def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p): return sha256_bytes(Path(p).read_bytes())


# ---------------------------------------------------------------------------
# Data: same cleaning and label as pytest_m2_standalone.py
# ---------------------------------------------------------------------------
def build_csv(raw_csv: str, out_csv: Path) -> dict:
    d = pd.read_csv(raw_csv, sep=";", decimal=",")
    d.columns = [str(c).strip() for c in d.columns]
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")].dropna(how="all")
    num = [c for c in d.columns if c not in ("Date", "Time")]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d[num] = d[num].replace(-200, np.nan)
    d["DateTime"] = pd.to_datetime(d["Date"].astype(str).str.strip() + " " +
                                   d["Time"].astype(str).str.strip().str.replace(".", ":", regex=False),
                                   dayfirst=True, errors="coerce")
    d = d.dropna(subset=["DateTime"]).reset_index(drop=True)
    d["Hour"], d["DayOfWeek"], d["Month"] = d["DateTime"].dt.hour, d["DateTime"].dt.dayofweek, d["DateTime"].dt.month
    for c in FEAT:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["CO(GT)", "NO2(GT)"]).reset_index(drop=True)
    for f in FEAT:
        d[f] = d[f].interpolate(method="linear", limit_direction="both")
        d[f] = d[f].fillna(d[f].median())
    d["Anomaly"] = (((d["CO(GT)"] > 4.0) | (d["NO2(GT)"] > 200.0)).astype(int))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    d[FEAT + ["Anomaly"]].to_csv(out_csv, index=False)
    return {"rows": len(d), "prevalence": round(float(d["Anomaly"].mean()), 4),
            "raw_sha256": sha256_file(raw_csv), "csv_sha256": sha256_file(out_csv)}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class OllamaLLM:
    def __init__(self, model, host="http://localhost:11434"):
        self.model, self.host = model, host

    def _post(self, path, payload, timeout):
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def check(self):
        try:
            tags = self._post("/api/show", {"model": self.model}, timeout=30)
        except Exception as e:
            sys.exit(f"Ollama check failed (is Ollama running and is '{self.model}' pulled?): {e}")
        return {"ollama_model_details": tags.get("details", {}), "ollama_modified_at": tags.get("modified_at")}

    def __call__(self, prompt):
        t0 = time.time()
        r = self._post("/api/chat", {"model": self.model, "stream": False,
                                     "messages": [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": prompt}],
                                     "options": {"temperature": TEMPERATURE, "num_predict": MAX_TOKENS}},
                       timeout=3600)
        return r["message"]["content"], {"prompt_tokens": r.get("prompt_eval_count", 0),
                                         "completion_tokens": r.get("eval_count", 0),
                                         "truncated": r.get("done_reason") == "length",
                                         "seconds": round(time.time() - t0, 1)}


class FakeLLM:
    model = "fake"

    def check(self):
        return {}

    def __call__(self, prompt):
        text = ("```python\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\n"
                "def load_data(file_path):\n    return pd.read_csv(file_path)\n"
                "def preprocess_data(data):\n    return data.drop(columns=['Anomaly']), data['Anomaly']\n"
                "def train_model(X_train, y_train):\n    m = RandomForestClassifier(n_estimators=10)\n"
                "    return m.fit(X_train, y_train)\n```\n"
                "```python\nfrom pipeline import load_data, preprocess_data, train_model\n"
                "def test_load():\n    assert len(load_data('data/air_quality.csv')) > 0\n"
                "def test_train():\n    X, y = preprocess_data(load_data('data/air_quality.csv'))\n"
                "    assert train_model(X[:200], y[:200]) is not None\n```\n"
                "```yaml\nname: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
                "    steps:\n      - uses: actions/checkout@v4\n      - run: pytest\n```\n")
        return text, {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(text) // 4,
                      "truncated": False, "seconds": 0.0}


# ---------------------------------------------------------------------------
# Extraction and execution (same block rules as pytest_m2_standalone.py)
# ---------------------------------------------------------------------------
def extract(text):
    blocks = re.findall(r"```(\w+)?\s*(.*?)```", text, re.DOTALL)
    pipeline = test = None
    parse, yaml_ok = [], None
    for lang, code in blocks:
        lang = (lang or "").lower()
        if lang in ("", "python", "py"):
            try:
                ast.parse(code); parse.append("OK")
            except SyntaxError as e:
                parse.append("FAIL:%s" % e.msg)
            if "def test_" in code:
                test = code
            elif "def load_data" in code or "def train_model" in code:
                pipeline = code
        elif lang in ("yaml", "yml"):
            try:
                import yaml; yaml.safe_load(code); yaml_ok = True
            except Exception:
                yaml_ok = False
    return pipeline, test, {"n_blocks": len(blocks), "python_parse": parse, "yaml_parses": yaml_ok}


def hint(tail):
    if re.search(r"NameError|ImportError|ModuleNotFoundError", tail): return "import_or_missing_name"
    if "AssertionError" in tail: return "assertion"
    if "FileNotFoundError" in tail: return "file_or_path"
    return "other"


def run_pytest(rd: Path, py: str):
    try:
        p = subprocess.run([py, "-m", "pytest", "test_pipeline.py", "-q", "--no-header", "-p", "no:cacheprovider"],
                           cwd=rd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return {"timeout": True, "all_pass": False, "failure_hint": "timeout"}
    out = p.stdout + p.stderr
    g = lambda pat: int(re.search(pat, out).group(1)) if re.search(pat, out) else 0
    tail = "\n".join(out.splitlines()[-25:])
    r = {"passed": g(r"(\d+) passed"), "failed": g(r"(\d+) failed"), "errors": g(r"(\d+) error"),
         "returncode": p.returncode, "tail": tail}
    r["all_pass"] = (p.returncode == 0 and r["passed"] > 0)
    if not r["all_pass"]:
        r["failure_hint"] = hint(tail)
    return r


def static_checks(code):
    if not code:
        return None
    try:
        ast.parse(code)
    except SyntaxError:
        return {"parses": False}
    rvh, iih = detect_rvh(code), detect_iih(code, PINNED_ENV)
    return {"parses": True, "is_rvh": rvh["is_rvh"], "rvh_violations": rvh["violations"],
            "is_iih": iih["is_iih"], "iih_violations": iih["violations"]}


def evaluate(text, rd: Path, csv: Path, py: str):
    (rd / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(csv, rd / "data" / "air_quality.csv")
    pipeline, test, parsed = extract(text)
    rec = {"extraction": parsed, "pytest": None,
           "static": {"pipeline": static_checks(pipeline), "test": static_checks(test)}}
    if pipeline and test:
        (rd / "pipeline.py").write_text(pipeline, encoding="utf-8")
        (rd / "test_pipeline.py").write_text(test, encoding="utf-8")
        rec["pytest"] = run_pytest(rd, py)
    else:
        rec["pytest"] = {"all_pass": False, "failure_hint": "blocks_not_extracted"}
    st = [s for s in rec["static"].values() if s and s.get("parses")]
    rec["is_rvh"] = any(s["is_rvh"] for s in st)
    rec["is_iih"] = any(s["is_iih"] for s in st)
    return rec


# ---------------------------------------------------------------------------
def main(a):
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"'{out}' already contains results; choose another --out, nothing was overwritten.")
    if a.backend == "replay":
        src = sorted(glob.glob(os.path.join(a.replay_dir, "run_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
        if not src:
            sys.exit(f"no run_* folders in {a.replay_dir}")
        llm, model, extra = None, a.replay_model, {"replay_dir": a.replay_dir}
    else:
        llm = OllamaLLM(a.model) if a.backend == "ollama" else FakeLLM()
        extra = llm.check()
        model = a.model if a.backend == "ollama" else "fake"
    if not Path(a.pytest_python).exists():
        sys.exit(f"pytest interpreter not found: {a.pytest_python} (pass --pytest-python)")
    out.mkdir(parents=True)
    data = build_csv(a.raw_csv, out / "air_quality.csv")
    env = subprocess.run([a.pytest_python, "-c", "import sys,pytest,mlflow,sklearn,pandas,numpy;"
                          "print(sys.version.split()[0],pytest.__version__,mlflow.__version__,"
                          "sklearn.__version__,pandas.__version__,numpy.__version__)"],
                         capture_output=True, text=True).stdout.split()
    meta = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "backend": a.backend, "model": model,
            "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "prompt_sha256": sha256_bytes(PROMPT.encode()),
            "system": SYSTEM, "data": data, "script_sha256": sha256_file(__file__),
            "generator_python": platform.python_version(),
            "pytest_env": dict(zip(["python", "pytest", "mlflow", "sklearn", "pandas", "numpy"], env)), **extra}
    (out / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    runs = []
    n = len(src) if a.backend == "replay" else a.n
    for k in range(1, n + 1):
        rd = out / f"run_{k:02d}"; rd.mkdir()
        if a.backend == "replay":
            text = Path(src[k - 1], "raw_llm_output.txt").read_text(encoding="utf-8")
            cost = {"replayed_from": src[k - 1]}
        else:
            text, cost = llm(PROMPT)
        (rd / "raw_llm_output.txt").write_text(text, encoding="utf-8")
        rec = {"run": k, "cost": cost, **evaluate(text, rd, out / "air_quality.csv", a.pytest_python)}
        (rd / "analysis.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        runs.append(rec)
        pt = rec["pytest"]
        print(f"run {k:2d}/{n}: pass={pt.get('all_pass')} ({pt.get('passed')} passed, {pt.get('failed')} failed) "
              f"{pt.get('failure_hint', '')} | RVH={rec['is_rvh']} IIH={rec['is_iih']} "
              f"| {cost.get('completion_tokens', '-')} tok, {cost.get('seconds', '-')} s"
              f"{' TRUNCATED' if cost.get('truncated') else ''}", flush=True)
    summary = {"n": n, "all_pass": sum(r["pytest"].get("all_pass", False) for r in runs),
               "extracted": sum(r["pytest"].get("failure_hint") != "blocks_not_extracted" for r in runs),
               "rvh": sum(r["is_rvh"] for r in runs), "iih": sum(r["is_iih"] for r in runs),
               "truncated": sum(bool(r["cost"].get("truncated")) for r in runs),
               "failure_hints": {h: sum(r["pytest"].get("failure_hint") == h for r in runs)
                                 for h in sorted({r["pytest"].get("failure_hint") for r in runs} - {None})}}
    (out / "results.json").write_text(json.dumps({"meta": meta, "summary": summary, "runs": runs}, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"saved -> {out / 'results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "replay", "fake"], required=True)
    ap.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model tag")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-dir", help="folder with run_*/raw_llm_output.txt (backend replay)")
    ap.add_argument("--replay-model", default="gpt-4o")
    ap.add_argument("--raw-csv", default=str(Path(__file__).resolve().parents[2] / "IJOSI Revision" / "Groupe 3" / "AirQualityUCI.csv"))
    ap.add_argument("--pytest-python", default=DEFAULT_PYTEST_PY)
    main(ap.parse_args())
