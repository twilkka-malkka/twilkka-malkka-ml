from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FOLDER = ROOT / "04_configs"
def load_config(config_name):
    config_path = CONFIG_FOLDER / config_name
    with open(config_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)