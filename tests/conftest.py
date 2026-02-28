import os
import sys
import logging

# Ensure src is in pythonpath so we can import 'config' module as the app does
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
os.environ.setdefault("DISCOVERY__EMBEDDING_PROFILE", "test-embedding")

from config import utils

_original_get_config_path = utils.get_config_path

def _mock_get_config_path(filename: str) -> str:
    try:
        return _original_get_config_path(filename)
    except SystemExit:
        # Map of config files to their example counterparts
        example_map = {
            "config.yaml": "config.example.yaml",
            "integrations.yaml": "integrations.example.yaml",
            "tools.yaml": "tools.example.yaml",
            "rbac.yaml": "rbac.example.yaml",
        }
        if filename in example_map:
            example_file = example_map[filename]
            logging.warning("%s not found, falling back to %s for testing", filename, example_file)
            return _original_get_config_path(example_file)
        raise

utils.get_config_path = _mock_get_config_path
# Set required environment variables to ensure config.py can import successfully
# during test collection, avoiding ValidationError on module level instantiation.
os.environ.setdefault("DISCOVERY__EMBEDDING_PROFILE", "test-embedding")