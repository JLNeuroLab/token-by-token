import pytest
from llm_project.models.ngrams.model import NGram

# This fixture provides a consistent, simple NGram model for all tests in this file.


@pytest.fixture
def simple_ngram_model():
    """A fixture that creates a simple trigram model from predictable text."""
    tokens = ["a", "b", "c", "a", "b", "d", "a", "b", "c"]
    model = NGram(tokens=tokens, n=3)
    model.build_all_ngram_freqs()
    return model


def test_ngram_creation(simple_ngram_model):
    """
    Tests if the model correctly creates n-grams of the specified order (n=3).
    """
    # ARRANGE
    expected_trigrams = [
        ("a", "b", "c"),
        ("b", "c", "a"),
        ("c", "a", "b"),
        ("a", "b", "d"),
        ("b", "d", "a"),
        ("d", "a", "b"),
        ("a", "b", "c"),
    ]

    # ACT
    actual_trigrams = list(simple_ngram_model.create_ngrams(3))

    # ASSERT
    assert actual_trigrams == expected_trigrams


def test_frequency_counting(simple_ngram_model):
    """
    Tests if the frequency counts for various n-grams are correct.
    """
    freqs = simple_ngram_model.ngram_freqs
    assert freqs.get(("a", "b", "c")) == 2
    assert freqs.get(("a", "b", "d")) == 1
    assert freqs.get(("a", "b")) == 3
    assert freqs.get(("a",)) == 3
    assert freqs.get(("d",)) == 1


def test_context_count_calculation(simple_ngram_model):
    """
    Tests if the get_context_count method correctly calculates and returns
    the frequency of different contexts.
    """
    # ARRANGE
    # For context ('a', 'b'), the trigrams are ('a', 'b', 'c') [2x] and ('a', 'b', 'd') [1x]. Total = 3.
    # For context ('b',), the bigrams are ('b', 'c') [2x] and ('b', 'd') [1x]. Total = 3.

    # ACT
    count_for_trigram_context = simple_ngram_model.get_context_count(("a", "b"))
    count_for_bigram_context = simple_ngram_model.get_context_count(("b",))

    # ASSERT
    assert count_for_trigram_context == 3
    assert count_for_bigram_context == 3
