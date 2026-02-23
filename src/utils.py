"""Utils
"""

import yaml
from typing import Dict
import json
from typing import List

class ConfigManager():

    def __init__(self, path : str):
        self.config = self._load_config(path)

    def _load_config(self, path : str) -> Dict:
        with open(path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)
        return config
    

def read_jsonl(path : str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {e}")
                continue
    return data

def write_jsonl(entries : List, path : str):
    with open(path, 'w') as outfile:
        for entry in entries:
            json.dump(entry, outfile)
            outfile.write('\n')

def create_batch_jsonl(path_templates : str, config : Dict):
    templates = read_jsonl(path_templates)
    entries = [{
            "custom_id" : str(i),
            "body" : {
                "max_tokens" : config["max_len"],
                "messages" : [
                    {
                        "role" : "user",
                        "content" : t["template"]
                    }
                ]
            }
        } for i,t in enumerate(templates)
    ]
    write_jsonl(entries, config["batch_input"])