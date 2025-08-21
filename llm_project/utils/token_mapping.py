import numpy as np

def token_id_mapping(tokens=None, ids=None, decode=False):
    """
    Maps tokens to ids, and ids to tokens. Returns a list of the mapped items.

    Args:
        tokens (list of str): tokenized text
        ids (list of int): list of integers
    Returns:
        token2id (dict): mapping from token to id
        id2token (dict): mapping from id to token
        mapped_list (list): list of ids (if encoding) or tokens (if decoding)
    """
    if tokens is None and not decode:
        raise ValueError("You must provide tokens when decode=False")

    # Extract unique token maintaining order of appearance
    unique_tokens = list(dict.fromkeys(tokens))
    token2id = {token: i for i, token in enumerate(unique_tokens)}
    id2token = {i: token for token, i in token2id.items()}
    
    # Decoding mode
    if decode:
        if ids is None:
            if tokens is None:
                raise ValueError("Provide either tokens or ids")
            ids = [token2id[token] for token in tokens]
        mapped_list = [id2token[id] for id in ids]
    # Encoding mode
    else:
        if tokens is None:
            raise ValueError("You should pass tokens when decoder=False")
        mapped_list = [token2id[token] for token in tokens]

    return token2id, id2token, mapped_list
