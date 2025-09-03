import os
def load_shakespeare(version ="raw"):
    version_map = {
        "raw": ("data/raw", "Shakespeare_clean_full.txt"),
        "train": ("data/processed", "Shakespeare_clean_train.txt"),
        "validation": ("data/processed", "Shakespeare_clean_valid.txt"),
        "test": ("data/processed", "Shakespeare_clean_test.txt"),
    }
    if version not in version_map:
        raise ValueError(f"version name {version} not valid, choose amongst {list(version_map.keys())}")

    folder, fname = version_map[version]

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    full_path = os.path.join(project_root, folder, fname)
    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read() 
    return text