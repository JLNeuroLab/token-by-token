from llm_project.utils.file_manager import load_item
from llm_project.bpe.bytepair_encoding import BPE
from llm_project.models.gpt.model import GPT
from llm_project.models.gpt.generator import Generator
import torch
import os


# --- Configuration ---
class GPTConfig:
    vocab_size = None
    block_size = 64
    embd_dim = 128
    n_layer = 4
    n_head = 4
    dropout = 0.1
    embd_pdrop = 0.1
    attn_pdrop = 0.1
    resid_pdrop = 0.1


# --- End of Config ---


def main():
    # Load Tokenizer
    print("Loading tokenizer...")
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    data_path = os.path.join(project_root, "data", "raw", "Shakespeare_clean_full.txt")
    bpe_results_folder = os.path.join("experiments", "bpe_results", "train_results")
    merges_file = "train_bpe_merges.pkl"
    vocab_file = "train_final_vocab.pkl"

    tokenizer = BPE(data_path, max_k=2000)
    try:
        tokenizer.merges = load_item(
            folder=bpe_results_folder, name=merges_file, base_dir=project_root
        )
        tokenizer.vocab = load_item(
            folder=bpe_results_folder, name=vocab_file, base_dir=project_root
        )
        tokenizer.build_token_mappings()
        print("  Tokenizer loaded successfully.")
    except FileNotFoundError:
        print("ERROR: Tokenizer files not found. Please run train.py first.")
        return
    # End Tokenizer Loading

    # Build Model
    config = GPTConfig()
    config.vocab_size = len(tokenizer.token_to_id)
    model = GPT(config)

    # Load Trained Weights
    print("Loading trained model weights...")
    model_path = os.path.join(project_root, "saved_models", "gpt_shakespeare.pth")
    try:
        # Load the state dict, making sure it's mapped to the correct device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("  Model weights loaded successfully.")
    except FileNotFoundError:
        print(
            f"ERROR: Model file not found at {model_path}. Please run train.py first."
        )
        return

    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Setup Generator
    generator = Generator(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.8,
        top_k=200,
    )

    # Generate Text
    prompt = "shall i compare thee to a summer's day?"
    print(f"\nPrompt: {prompt}")

    input_tokens = tokenizer.encode(prompt)

    generated_token_ids = generator.generate(input_tokens)

    output_text = tokenizer.decode(generated_token_ids)

    print("\nGenerated Text:")
    print(output_text)


if __name__ == "__main__":
    main()
