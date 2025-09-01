import os
import pytest
from llm_project.models.ngrams.trainer import NGramTrainer
from llm_project.bpe.bytepair_encoding import BPE
from llm_project.models.ngrams.model import NGram

# This fixture creates a temporary directory for each test function,
# so that saved files from one test don't interfere with another.


@pytest.fixture
def ngram_trainer_setup(tmp_path):
    """Sets up an NGramTrainer instance with a temporary directory as its root."""
    return NGramTrainer(model=None, tokens=None, n=3, max_k=10, root=tmp_path)


def test_train_save_load_model(ngram_trainer_setup):
    """
    Integration test: Trains, saves, and loads a model, verifying its state.
    """
    trainer1 = ngram_trainer_setup

    # ARRANGE: Provide simple training text and a pre-made BPE object
    trainer1.tokens = ["a", "b", "c", "a", "b", "d"]
    trainer1.bpe = BPE(max_k=10, text="abcabd")
    trainer1.bpe.BPE_encoder()  # Manually run encoder to generate merges

    # ACT 1: Train and save
    model1, merges1 = trainer1.train(force_retrain=True, tune_lambdas=False)

    # ASSERT 1: Check that files were created
    model_dir = os.path.join(trainer1.root, "experiments", "saved_models", "ngram")
    assert os.path.exists(os.path.join(model_dir, "ngram_model_n3_k10.pkl"))
    assert os.path.exists(os.path.join(model_dir, "BPE_merges_k10.pkl"))

    # ACT 2: Create a new trainer and load the saved model
    trainer2 = NGramTrainer(model=None, tokens=None, n=3, max_k=10, root=trainer1.root)
    model2, merges2 = trainer2.train(force_retrain=False)  # Should load from disk

    # ASSERT 2: Check that the loaded model state is correct
    assert model2 is not None
    assert model2.n == model1.n
    assert len(model2.tokens) == len(model1.tokens)
    assert set(model2.ngram_freqs.keys()) == set(model1.ngram_freqs.keys())


def test_compute_perplexity_sanity_check(ngram_trainer_setup):
    """
    This test confirms the perplexity logic is correct. A low (but not 1.0)
    score on a predictable text is expected. A very high number indicates a bug.
    """
    trainer = ngram_trainer_setup

    # ARRANGE: Create a perfectly repetitive text
    simple_text = "a b c a b c a b c a b c"
    trainer.bpe = BPE(max_k=10, text=simple_text)
    trainer.bpe.BPE_encoder()
    trainer.tokens = trainer.bpe.tokens

    # Train the model on this simple text
    trainer.model, _ = trainer.train(force_retrain=True, tune_lambdas=False)

    # ACT: Compute perplexity on the SAME text it was trained on
    validation_tokens = trainer.bpe.BPE_segmenter(simple_text)
    default_lambdas = [1 / trainer.n] * trainer.n
    perplexity = trainer.compute_perplexity(validation_tokens, lambdas=default_lambdas)

    print(f"Sanity Check Perplexity: {perplexity}")

    # ASSERT: Perplexity on a known text should be low.
    # We expect a value around 2.5, so < 3.0 is a robust check.
    assert perplexity < 3.0
