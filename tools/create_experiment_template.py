import os
import argparse

def create_experiment_folders(exp_name):
    base_path = os.path.join("experiments", exp_name)
    subfolders = [
        "data",
        "logs",
        "scripts",
        "models",
        "prompts",
        "output"
    ]

    for folder in subfolders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok = True)
        print(f"Created: {path}")

    # Add README.md
    readme_path = os.path.join(base_path, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(f"# Experiment: {exp_name}\n\nDescribe this experiment here.\n")
        print(f"Created: {readme_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create experiment folder template.")
    parser.add_argument("exp_name", help="Name of the experiment folder (inside /experiments)")
    args = parser.parse_args()

    create_experiment_folders(args.exp_name)