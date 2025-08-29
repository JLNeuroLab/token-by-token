import os
import shutil
import psutil
from llm_project.bpe.bytepair_encoding import BPE, print_resource_usage


def test_bpe_encode_decode_round_trip():
    """
    Tests that encoding a string and then decoding it returns the
    original string.
    """
    # ARRANGE
    training_text = "ab ab ab ac"
    bpe = BPE(max_k=100, text=training_text)
    bpe.BPE_encoder()
    original_string = "ab ac"

    # ACT
    encoded_ids = bpe.encode(original_string)
    decoded_string = bpe.decode(encoded_ids)

    # ASSERT
    assert decoded_string == original_string


def test_bpe_selects_most_frequent_pair():
    """
    Tests that the BPE algorithm correctly identifies and merges the most
    frequent pair of tokens.
    """
    # ARRANGE
    training_text = "aaababa"
    bpe = BPE(max_k=100, text=training_text)

    # ACT
    bpe.BPE_encoder()
    first_merge_rule = bpe.merges[0]

    # ASSERT
    assert first_merge_rule == (("a", "a"), "aa")


def test_bpe_handles_unknown_characters():
    """
    Tests that the tokenizer correctly handles characters that were not
    present in the training data by assigning them an ID of -1.
    """
    # ARRANGE
    training_text = "aaaaa"
    bpe = BPE(max_k=50, text=training_text)
    bpe.BPE_encoder()

    test_string_with_unknowns = "abac"

    # ACT
    encoded_ids = bpe.encode(test_string_with_unknowns)

    # ASSERT
    id_of_a = bpe.token_to_id["a"]
    assert encoded_ids == [id_of_a, -1, id_of_a, -1]


def test_bpe_high_k_resource_usage():
    """
    Track system usage during BPE training (high merge count).
    """
    training_text = " ".join(["abc def ghi"] * 10000)
    step_resources = {}

    # Custom resource tracker function
    def track_resource_usage(self, step: int):
        ram_used = psutil.virtual_memory().used / 1024**2  # MB
        cpu_percent = psutil.cpu_percent(interval=0.1)
        total, used, _ = shutil.disk_usage(os.getcwd())
        disk_used = used / 1024**3  # GB

        step_resources[step] = {
            "ram_MB": round(ram_used, 2),
            "cpu_percent": round(cpu_percent, 2),
            "disk_GB": round(disk_used, 2),
        }

        return 0  # Dummy duration for compatibility

    # Pass your custom function to BPE
    bpe = BPE(max_k=900, text=training_text, track_resource_fn=track_resource_usage)

    bpe.BPE_encoder()

    print("\n[BPE RESOURCE TRACKING]")
    for step, data in step_resources.items():
        print(
            f"[STEP {step:03}] RAM: {data['ram_MB']} MB | CPU: {data['cpu_percent']}% | Disk: {data['disk_GB']} GB"
        )

    assert len(step_resources) > 0
