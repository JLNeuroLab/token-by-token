import numpy as np
from llm_project.models.gpt.generator import Generator


class MockModel:
    pass


class MockTokenizer:
    pass


def test_softmax_produces_valid_probability_distribution():
    """
    Tests if the softmax function correctly turns logits into a probability
    distribution that sums to 1.
    """
    # ARRANGE
    generator = Generator(model=MockModel(), tokenizer=MockTokenizer())
    logits = np.array([1.0, 2.0, 3.0, 4.0])

    # ACT
    probabilities = generator.softmax(logits)

    # ASSERT
    assert np.isclose(np.sum(probabilities), 1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_top_k_sample_selects_from_correct_indices():
    """
    Tests that top_k_sample only returns a token ID from the set of
    the k most likely tokens.
    """
    # ARRANGE
    generator = Generator(model=MockModel(), tokenizer=MockTokenizer(), top_k=3)
    logits = np.array([1.0, 0.5, 8.0, 2.0, 3.0, 9.0, 0.1, 4.0, 8.5])
    expected_top_indices = {2, 5, 8}

    # ACT
    chosen_token_id = generator.top_k_sample(logits)

    # ASSERT
    assert chosen_token_id in expected_top_indices


def test_top_p_sample_selects_from_nucleus():
    """
    Tests that top_p_sample only returns a token ID from the nucleus,
    the smallest set of tokens whose cumulative probability exceeds top_p.
    """
    # ARRANGE
    generator = Generator(model=MockModel(), tokenizer=MockTokenizer(), top_p=0.9)
    logits = np.log(np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]))
    expected_nucleus_indices = {0, 1, 2, 3, 4}

    # ACT
    chosen_token_id = generator.top_p_sample(logits)

    # ASSERT
    assert chosen_token_id in expected_nucleus_indices
