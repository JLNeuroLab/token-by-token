import os
import pickle
import matplotlib.pyplot as plt

# ------------------------Method to automatically save results------------------------------
def save_item(item, folder : str , name : str, text_version=True, base_dir = None):
    """
    Save different type of files and creates an according folder if it does not exist
    
        Args:
            item: type of file (e.g., string, list, dictionary, matplotlib figure)
            folder: name of the folder in which the file is saved in the form of string
            name: name that we assign to the file
            text_version: parameter that saves some files in a reading format if set to true
            base_dir: takes the base directory of the current file
    
    """
    if base_dir is None:
        base_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), 
                    "..", 
                    "..")
                )

    folder = os.path.join(base_dir, folder)
    os.makedirs(folder, exist_ok=True)
    output_file = os.path.join(folder, name)
    if isinstance(item, str):
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(item)
        print(f"text normalized and saved: {os.path.basename(folder)}/{name}")

    elif isinstance(item, plt.Figure):
        item.savefig(output_file)
        print(f"plot saved in: {os.path.basename(folder)}/{name}")

    elif isinstance(item, (list, tuple, set, dict)):
        with open(output_file, "wb") as f:
            pickle.dump(item, f)
        print(f"data saved in: {os.path.basename(folder)}/{name}")

        if text_version is True:
            text_file = os.path.splitext(output_file)[0] + ".txt"
            with open(text_file, "w", encoding="utf-8") as f:

                if isinstance(item, dict):
                 
                    for k, v in item.items():
                        f.write(f"{k}\t{v}\n")
                else:
                    for elem in item:
                        if isinstance(elem, tuple):
                            
                            f.write("\t".join(map(str, elem)) + "\n")
                        else:
                            f.write(str(elem) + "\n")
                            
            print(f"data saved in readable text: {os.path.basename(folder)}/{os.path.basename(text_file)}")
    else:
        raise TypeError(f"Unsupported type: {type(item)}")
    
    
def load_item(folder: str, name: str, base_dir=None):
    """
    Load an item previously saved with save_item.

    Args:
        folder (str): folder where the item was saved
        name (str): filename
        base_dir (str|None): optional base directory

    Returns:
        item: the loaded object (str, dict, list, tuple, set, etc.)
    """
    if base_dir is None:
        base_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                ".."
            )
        )

    path = os.path.join(base_dir, folder, name)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found in folder {folder}")

    ext = os.path.splitext(name)[1].lower()
    
    if ext in [".txt"]:  # assume plain text file
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    elif ext in [".pkl", ".dat", ".bin"]:  # assume pickled object
        with open(path, "rb") as f:
            return pickle.load(f)
    
    elif ext in [".png", ".jpg", ".jpeg", ".pdf"]:  # assume saved figure
        return plt.imread(path)
    
    else:
        raise TypeError(f"Unsupported file type: {ext}")
