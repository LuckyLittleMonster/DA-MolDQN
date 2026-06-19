import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra

from src.entry import run_entry

_MODE = "testing"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    run_entry(cfg, mode="testing" if "testing" != "testing" else "test")


if __name__ == "__main__":
    main()
