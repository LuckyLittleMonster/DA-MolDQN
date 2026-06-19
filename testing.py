import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hydra

from src.entry import run_entry

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    run_entry(cfg, mode="test")


if __name__ == "__main__":
    main()
