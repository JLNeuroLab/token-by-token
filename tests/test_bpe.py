from llm_project.bpe.bytepair_encoding import BPE


def test_bpe_encode_decode_round_trip():
    """
    Tests that encoding a string and then decoding it returns the
    original string.
    """
    # ARRANGE
    training_text = "ab ab ab ac"
    bpe = BPE(max_k=1, text=training_text)
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
    bpe = BPE(max_k=1, text=training_text)

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
    bpe = BPE(max_k=5, text=training_text)
    bpe.BPE_encoder()

    test_string_with_unknowns = "abac"

    # ACT
    encoded_ids = bpe.encode(test_string_with_unknowns)

    # ASSERT
    id_of_a = bpe.token_to_id["a"]
    assert encoded_ids == [id_of_a, -1, id_of_a, -1]
