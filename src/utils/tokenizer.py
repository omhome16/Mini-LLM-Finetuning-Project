from transformers import AutoTokenizer


def load_pythia_tokenizer(model_name: str = "EleutherAI/pythia-70m"):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Pythia's tokenizer doesn't have a dedicated padding token, so we set it to the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer