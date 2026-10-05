from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


"""Encoder/tokenizer loading and hidden-state layer specifications."""


def load_models(model_type, sentence=False):
    if model_type == "scibert":
        spec = "allenai/scibert_scivocab_cased"
    elif model_type == "biobert":
        spec = "dmis-lab/biobert-base-cased-v1.2"
    elif model_type == "pubmedbert":
        spec = "neuml/pubmedbert-base-embeddings"
    elif model_type == "biobert-stsb":
        spec = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    elif model_type == "bert":
        spec = "google-bert/bert-base-uncased"
    elif model_type == "bluebert":
        spec = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    else:
        raise ValueError(f"{model_type} unsupported")
    if sentence:
        tokenizer, model = None, SentenceTransformer(spec)
    else:
        tokenizer, model = (
            AutoTokenizer.from_pretrained(spec),
            AutoModel.from_pretrained(spec),
        )
    return tokenizer, model


def normalize_layers_spec(
    layers_spec: str | list[int],
    *,
    n_hidden_states: int | None = None,
) -> list[int]:
    """Parse and validate indices for the stacked ``hidden_states`` tensor.

    String form follows the same convention as historical :func:`str2layers`: each
    digit is a distinct layer counted from the end, e.g. ``\"1\"`` → ``[-1]``,
    ``\"12\"`` → ``[-2, -1]``. Commas in the string are ignored.

    Args:
        layers_spec: Digit-only string or list of **negative** indices (HF convention).
        n_hidden_states: If set (length of first dim of stacked hidden states), indices
            must satisfy ``layer >= -n_hidden_states``.

    Returns:
        Sorted unique negative layer indices.

    Raises:
        ValueError: Empty spec, positive indices, ``\"sent\"``, or out-of-range indices.
    """
    if isinstance(layers_spec, str):
        spec = layers_spec.strip()
        if not spec:
            raise ValueError("layers_spec string is empty")
        if spec == "sent":
            raise ValueError(
                "layers_spec 'sent' is not valid for transformer hidden-state pooling"
            )
        if "," in spec:
            spec = "".join(spec.split(","))
        if not spec.isdigit():
            raise ValueError(
                f"layers_spec string must be digits only (e.g. '1' or '12'), got {layers_spec!r}"
            )
        layers = sorted({-abs(int(ch)) for ch in spec})
    else:
        if not layers_spec:
            raise ValueError("layers_spec list is empty")
        for i, layer in enumerate(layers_spec):
            if layer >= 0:
                raise ValueError(
                    f"layer index must be negative (HF hidden_states convention), got {layer} at position {i}"
                )
        layers = sorted(set(layers_spec))
    if n_hidden_states is not None:
        for layer in layers:
            if layer < -n_hidden_states:
                raise ValueError(
                    f"layer {layer} is out of range for {n_hidden_states} stacked hidden states"
                )
    return layers


def str2layers(layers_spec: str | list[int]) -> list[int]:
    """Parse layer specification; same rules as :func:`normalize_layers_spec`."""
    return normalize_layers_spec(layers_spec, n_hidden_states=None)
