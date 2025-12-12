import re 
import os
import sys
import time
import socket
import argparse
import subprocess

from datetime import datetime
def main(experiment_config_filename, skip_build=False):
    config_data = parse_toml(experiment_config_filename)

    if not config_data:
        print()
        print(colored("ERROR: Configuration data is empty.", "red"))
        sys.exit(1)
    
    # Add skip_build to settings if provided via command line
    if skip_build:
        if 'settings' not in config_data:
            config_data['settings'] = {}
        config_data['settings']['skip_build'] = True
    
    run_experiment(config_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a seismic experiment on a dataset and query it.")
    parser.add_argument("--exp", required=True, help="Path to the experiment configuration TOML file.")
    parser.add_argument("--skip-build", action="store_true", help="Skip the index building step entirely.")
    args = parser.parse_args()

    main(args.exp, args.skip_build)
    sys.exit(0)