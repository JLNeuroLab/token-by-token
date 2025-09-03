import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

def get_project_root(marker="llm_project"):
    """
    Trova la root del progetto cercando la cartella 'marker'.
    Sale nei genitori fino a trovarla.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root not found. Couldn't locate folder '{marker}'")

def get_model_path(root, category, subdir=None, final=False):
    """
    Create and return a standardized path for experiment assets.

    Args:
        root (str): Root folder of the project.
        category (str): Asset type (e.g., "saved_models", "plots").
        name (str, optional): Subfolder for a specific model or experiment.
        ensure_exists (bool): If True, create the folder if it doesn't exist.

    Returns:
        str: Full path to the folder.
    """
    if root is None:
        root = get_project_root()
    root = Path(root)
    base_folder = "experiments" if not final else "saved_models"
    exp_path = root / base_folder / category
    if subdir:
        exp_path = exp_path / subdir
    print(f"folder: {exp_path}")
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path


def save_model(model, root, category="models", subdir=None, filename=None, final=False):
    """
    Save an object (model or state dictionary) to a structured folder under root.

    Args:
        obj: Object to save (model or state dict).
        root (str): Root folder of the project.
        category (str): Asset type folder (default: "saved_models").
        name (str, optional): Subfolder for specific experiment or model.
        filename (str, optional): File name. Defaults to 'model.pkl'.

    Returns:
        str: Full path to the saved file.
    """
    # Build folder path
    save_dir = get_model_path(root, category, subdir, final=final)
    if filename is None:
        filename = "model.pkl"
    file_path = os.path.join(save_dir, filename)

    os.makedirs(save_dir, exist_ok=True)

    # Save with pickle
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    return file_path

def load_model(root, filename, category="models", final=None):
    """
    Load a previously saved model or state dictionary.

    Args:
        file_path (str): Path to the saved pickle file.

    Returns:
        object: Loaded object (model or state dictionary).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    file_path = get_model_path(root, category, final=final) / filename

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found in: {file_path}")

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    return model

def save_tokenizer(bpe, root, filename, category="tokenizers", final=False):
    """
    Save only the BPE tokenizer (merges) to a structured folder under root.

    Args:
        bpe (BPE): A trained BPE object.
        root (str): Root folder of the project.
        category (str): Asset type folder (default: "saved_models").
        name (str, optional): Subfolder for a specific experiment or model.
        filename (str, optional): File name for the merges.

    Returns:
        str: Full path to the saved file.
    """
    data = {
        "bpe": bpe,
        "tokens": getattr(bpe, "tokens", None)
    }
    return save_model(
        model=data,
        root=root,
        category=category,
        filename=filename,
        final=final
    )


def load_tokenizer(root, filename, category="tokenizers", final=False):
    """
    Load only the BPE tokenizer from a saved state dictionary.

    Args:
        file_path (str): Path to the saved pickle file containing BPE merges.

    Returns:
        BPE: A BPE object with merges loaded and ready to tokenize text.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no BPE merges are found in the file.
    """
    data = load_model(root=root, filename=filename, category=category, final=final)
    return data["bpe"], data.get("tokens", None)


