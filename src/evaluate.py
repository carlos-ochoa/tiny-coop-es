"""Evaluates a set of generated data using LLM as a judge
"""

import mlflow
from src.utils import ConfigManager, read_jsonl
from src.metrics import scorers
import pandas as pd

config = ConfigManager("config.yaml").config

if config["evaluation"]["metrics"][0] == "all":
    scorers = list(scorers.values())
else:
    scorers = [scorers[s] for s in config["evaluation"]["metrics"]]

data = []
data = read_jsonl(config["data"]["output"])
inputs = read_jsonl(config["data"]["prompt_templates"])

mlflow.set_experiment("Story Generation")

if config["models"]["generator_provider"] == "anthropic":
    eval_dataset = pd.DataFrame({
    "inputs": [
            {"text_id": i} 
            for i in range(len(data))
        ],
        "outputs" : [
            d["text"] for d in data
        ]
    })
elif config["models"]["generator_provider"] == "mistral":
    eval_dataset = pd.DataFrame({
    "inputs": [
            {"text_id": i} 
            for i in range(len(data))
        ],
        "outputs" : [
            d["response"]["body"]["choices"][0]["message"]["content"] for d in data
        ]
    })

with mlflow.start_run(run_name=config["evaluation"]["run_name"]):
    mlflow.genai.evaluate(
        data=eval_dataset,
        scorers=scorers,
    )
