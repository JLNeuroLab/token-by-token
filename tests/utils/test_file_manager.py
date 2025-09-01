import os
import pickle
import pytest
from llm_project.utils.file_manager import save_item, load_item

# Pytest fixture to create a temporary directory for test files


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


def test_save_and_load_dict_round_trip(temp_dir):
    """
    Tests that a dictionary can be saved and loaded back,
    and that the result is identical to the original.
    """
    # ARRANGE
    original_dict = {("a", "b"): 10, ("b", "c"): 5}
    file_name = "test_dict.pkl"

    # ACT
    save_item(original_dict, temp_dir, file_name)
    loaded_dict = load_item(temp_dir, file_name)

    # ASSERT
    assert os.path.exists(os.path.join(temp_dir, file_name))
    assert original_dict == loaded_dict


def test_save_and_load_list_round_trip(temp_dir):
    """
    Tests that a list can be saved and loaded back perfectly.
    """
    # ARRANGE
    original_list = [("a", "b", "c"), ("d", "e", "f")]
    file_name = "test_list.pkl"

    # ACT
    save_item(original_list, temp_dir, file_name)
    loaded_list = load_item(temp_dir, file_name)

    # ASSERT
    assert original_list == loaded_list


def test_save_text_file(temp_dir):
    """
    Tests that a string is correctly saved as a .txt file.
    """
    # ARRANGE
    original_text = "hello world"
    file_name = "test.txt"

    # ACT
    save_item(original_text, temp_dir, file_name)
    loaded_text = load_item(temp_dir, file_name)

    # ASSERT
    assert original_text == loaded_text


def test_load_nonexistent_file_raises_error(temp_dir):
    """
    Tests that load_item raises a FileNotFoundError for a missing file.
    """
    # ARRANGE / ACT / ASSERT
    with pytest.raises(FileNotFoundError):
        load_item(temp_dir, "nonexistent.pkl")
