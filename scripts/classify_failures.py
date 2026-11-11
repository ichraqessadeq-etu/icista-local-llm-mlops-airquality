# -*- coding: utf-8 -*-
"""Failure taxonomy of the ICISTA study, from the saved pytest outputs (no model call).

Each failing generation gets ONE category, by priority:
  mlflow_api    a call to an MLflow API that does not exist in the pinned version
                (ImportError from mlflow..., or an unexpected keyword argument of an MLflow function)
  missing_name  a name used without being imported or defined (NameError), mostly in the test file
  data_logic    the code runs but mishandles the data (KeyError on a column, NaN, wrong return value...)
  collection    pytest cannot even collect the test file (syntax or import error at collection)
    py -3.13 classify_failures.py   -> failure_categories.json
"""
import glob, json, re

RUNS = [("gpt-4o", "runs_gpt4o_replay"), ("qwen2.5-coder:3b", "runs_qwen25coder3b"),
        ("qwen2.5-coder:7b", "runs_qwen25coder7b"), ("llama3.1:8b", "runs_llama31_8b"), ("mistral:7b", "runs_mistral7b")]


def category(tail):
    if re.search(r"ImportError: cannot import name '[^']+' from 'mlflow", tail) or \
       re.search(r"(start_run|log_model|log_param|log_metric)\(\) got an unexpected keyword", tail):
        return "mlflow_api"
    if re.search(r"NameError: name '[^']+' is not defined|name '[^']+' is not defined", tail):
        return "missing_name"
    if re.search(r"KeyError|ValueError|TypeError|AssertionError|AttributeError|FileNotFoundError", tail):
        return "data_logic"
    return "collection"


out = {}
for model, root in RUNS:
    cats = {"pass": 0, "mlflow_api": 0, "missing_name": 0, "data_logic": 0, "collection": 0}
    for d in sorted(glob.glob(f"{root}/run_*")):
        p = json.load(open(f"{d}/analysis.json", encoding="utf-8"))["pytest"]
        cats["pass" if p.get("all_pass") else category(p.get("tail") or "")] += 1
    out[model] = cats
    print(f"{model:18s} {cats}")
json.dump(out, open("failure_categories.json", "w", encoding="utf-8"), indent=2)
