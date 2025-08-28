from llm_project.bpe.bytepair_encoding import BPE
from llm_project.models.gpt.model import GPT
import torch
from torch.nn import functional as F
from llm_project.utils.file_manager import save_item, load_item
import os
# import sys
#
# Add the project root to the Python path
# sys.path.insert(
#     0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# )


# --- Configuration ---
BATCH_SIZE = 32
BLOCK_SIZE = 64
MAX_ITERS = 8000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
# --- End of Config ---

# Device Setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")
device = torch.device(device)


# Data Loading and Preparation
data_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "data",
    "raw",
    "Shakespeare_clean_full.txt",
)
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

print("Training tokenizer...")
tokenizer = BPE(data_path, max_k=2000)

# tokenizer.load_and_normalize()
# tokenizer.BPE_encoder()
#
# vocab_list = list(tokenizer.vocab.keys())
# token_to_id = {token: i for i, token in enumerate(vocab_list)}
# id_to_token = {i: token for token, i in token_to_id.items()}

# Caching Logic for saving time on debbug
# project_root = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "..", "..", "..")
# )

project_root = os.getcwd()

bpe_results_folder = os.path.join(
    "experiments", "bpe_results", "train_results")
merges_file = "train_bpe_merges.pkl"
vocab_file = "train_final_vocab.pkl"

try:
    # Try to load the pre-saved merges and vocab
    tokenizer.merges = load_item(
        folder=bpe_results_folder, name=merges_file, base_dir=project_root
    )
    tokenizer.vocab = load_item(
        folder=bpe_results_folder, name=vocab_file, base_dir=project_root
    )
    print(f">>Loaded pre-trained tokenizer from {bpe_results_folder}")
    tokenizer.build_token_mappings()

except FileNotFoundError:
    # If files don't exist, train the tokenizer from scratch
    print(">>Training tokenizer from scratch...")
    tokenizer.load_and_normalize()
    tokenizer.text = text  # The BPE class needs the full text to train on
    tokenizer.BPE_encoder()

    # And save the results using the same paths for next time
    print(f">>Saving tokenizer to {bpe_results_folder}...")
    save_item(
        tokenizer.merges,
        folder=bpe_results_folder,
        name=merges_file,
        base_dir=project_root,
    )
    save_item(
        tokenizer.vocab,
        folder=bpe_results_folder,
        name=vocab_file,
        base_dir=project_root,
    )


# def encode(s):
#     return [
#         token_to_id.get(t, -1) for t in tokenizer.BPE_segmenter(s)
#     ]  # Encoder: string -> list[int]
#
#
# def decode(l):
#     return "".join([id_to_token.get(i, "") for i in l]).replace(
#         "_", " "
#     )  # Decoder: list[int] -> string
#

vocab_size = len(tokenizer.token_to_id)
print(f"Vocabulary size: {vocab_size}")

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# first 90% will be train, rest val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data Batching
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i: i + BLOCK_SIZE] for i in ix])
    # The target sequence is shifted by one position
    y = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# Model and Optimizer
class GPTConfig:
    vocab_size = vocab_size
    block_size = BLOCK_SIZE
    embd_dim = 128
    n_layer = 4
    n_head = 4
    dropout = 0.2
    embd_pdrop = 0.2
    attn_pdrop = 0.2
    resid_pdrop = 0.2


config = GPTConfig()
model = GPT(config)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Scaler for mixed precision training on CUDA (Automatic mixed Precision)
use_amp = device == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
# End Model and Optimizer


# --- Evaluation Function ---
@torch.no_grad()  # no need to calculate gradients here
def estimate_loss():
    out = {}
    model.eval()  # set model to eval mode
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(X)
                B, T, C = logits.shape
                logits_view = logits.view(B * T, C)
                Y_view = Y.view(B * T)
                loss = F.cross_entropy(logits_view, Y_view)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  # set model back to train mode
    return out


# End Evaluation Function


# Training Loop
print("Starting training...")
for iter in range(MAX_ITERS):
    # evaluate loss periodically
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # get a batch of data
    xb, yb = get_batch("train")

    # calculate the loss
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        yb = yb.view(B * T)
        loss = F.cross_entropy(logits, yb)

    # backpropagation
    optimizer.zero_grad(set_to_none=True)  # clear old gradients
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print("Training finished.")
# End Training Loop


# Save the Model
<< << << < HEAD
# save_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "saved_models")
save_path = os.path.join("saved_models")
== == == =
save_path = os.path.join(os.path.dirname(
    __file__), "..", "..", "..", "saved_models")
>>>>>> > origin/jacopo_branch
os.makedirs(save_path, exist_ok=True)
model_save_path = os.path.join(save_path, "gpt_shakespeare.pth")
print(f"Saving model to {model_save_path}")
torch.save(model.state_dict(), model_save_path)
