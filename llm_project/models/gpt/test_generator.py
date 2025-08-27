import sys
import os

# Patch sys.path BEFORE any project-specific imports
# Add llm_project/ to sys.path

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
# ========== All imports below are moved inside main ==========
if __name__ == "__main__":
    import torch
    import re
    from llm_project.models.gpt.generator import Generator
    from llm_project.models.gpt.model import GPT
    from llm_project.bpe.bytepair_encoding import BPE  # BPE tokenizer

    # ========== Subclass BPE to act like a tokenizer ==========
    class MyTokenizer(BPE):
        def build_token_mappings(self):
            vocab_list = list(self.vocab.keys())
            if "<UNK>" not in vocab_list:
                vocab_list.insert(0, "<UNK>")

            self.token_to_id_map = {token: i for i, token in enumerate(vocab_list)}
            self.id_to_token_map = {
                i: token for token, i in self.token_to_id_map.items()
            }
            self.unk_token_id = self.token_to_id_map["<UNK>"]

        def encode(self, text):
            # Normalize text like training

            text = text.lower().strip()
            text = text.replace(" ", "_")
            text = re.sub(r"[^\w\s?!.']", "", text)
            tokens = self.BPE_segmenter(text)

            # Convert to IDs, skip unknowns
            token_ids = [self.token_to_id_map.get(t, self.unk_token_id) for t in tokens]
            return token_ids

        def decode(self, token_ids):
            tokens = [self.id_to_token_map.get(i, "<UNK>") for i in token_ids]
            return "".join(tokens).replace("_", " ")

        @property
        def vocab_size(self):
            return len(self.token_to_id_map)

    # ========== Temporary Config class ==========
    class Config:
        vocab_size = None
        block_size = 64
        embd_dim = 128
        n_layer = 4
        n_head = 4
        dropout = 0.1
        embd_pdrop = 0.1
        attn_pdrop = 0.1
        resid_pdrop = 0.1

    # ========== Run generation ==========
    def main():
        # Initialize and train tokenizer
        tokenizer = MyTokenizer("data/raw/Shakespeare_clean_full.txt", max_k=2000)
        tokenizer.load_and_normalize()
        test_text, train_text = tokenizer.split_train_test()
        tokenizer.train_text = train_text[:10000]
        tokenizer.test_text = test_text[:1000]
        tokenizer.text = tokenizer.train_text
        tokenizer.BPE_encoder()
        tokenizer.build_token_mappings()

        # Build GPT model
        config = Config()
        config.vocab_size = tokenizer.vocab_size
        model = GPT(config)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        # Setup Generator
        generator = Generator(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=1.0,
            top_k=10,
            top_p=1.0,
            eos_token_id=None,
            mode="top_k",
        )

        # Prompt → Generate → Output
        prompt = "Shall I compare thee"
        input_tokens = tokenizer.encode(prompt)
        # debug check.....
        print("Input tokens:", input_tokens)
        print("Token types:", set(type(t) for t in input_tokens))
        generated_tokens = generator.generate(input_tokens)
        output_text = tokenizer.decode(generated_tokens)

        print("\nPrompt:", prompt)
        print("enerated:", output_text)

    main()
