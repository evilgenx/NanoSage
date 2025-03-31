# config_utils.py
import os
import yaml

def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.isfile(config_path):
        print(f"[WARN] Config file not found: {config_path}. Returning empty config.")
        return {}

    config = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded_data = yaml.safe_load(f)
            # Ensure we return a dict even if the file is empty or contains only 'null'
            config = loaded_data if isinstance(loaded_data, dict) else {}
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file {config_path}: {e}")
        # Return empty dict on parsing error
        config = {}
    except FileNotFoundError:
        # This case is already handled by the initial check, but good practice
        print(f"[ERROR] File not found during open: {config_path}")
        config = {}
    except Exception as e:
        # Catch other potential file reading errors
        print(f"[ERROR] Failed to read config file {config_path}: {e}")
        config = {}

    return config

def save_config(config_path, config_data):
    """Saves configuration data to a YAML file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"[INFO] Configuration saved to: {config_path}")
        return True
    except yaml.YAMLError as e:
        print(f"[ERROR] Error writing YAML file {config_path}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to save config file {config_path}: {e}")
        return False
