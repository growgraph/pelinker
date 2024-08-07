# pylint: disable=E1120

import re
from string import punctuation, whitespace
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pandas as pd

from sklearn.metrics import accuracy_score

MAX_LENGTH = 512


def load_models(model_type, sentence=False):
    if model_type == "scibert":
        spec = "allenai/scibert_scivocab_cased"
    elif model_type == "biobert":
        spec = "dmis-lab/biobert-base-cased-v1.2"
    elif model_type == "pubmedbert":
        spec = "neuml/pubmedbert-base-embeddings"
    elif model_type == "biobert-stsb":
        spec = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
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


def text_to_tokens_embeddings(texts: list[str], tokenizer, model):
    encoding = tokenizer.batch_encode_plus(
        texts,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {k: encoding[k] for k in ["input_ids", "token_type_ids", "attention_mask"]}

    with torch.no_grad():
        outputs = model(output_hidden_states=True, **inputs)

    # n_layers x nb x n_len x n_emb
    tt = torch.stack(outputs.hidden_states)

    # nb x n_len x 2
    offsets = encoding["offset_mapping"]

    # fill with zeros latent vectors for padded tokens
    mask = encoding["attention_mask"]
    mask = mask.unsqueeze(-1).unsqueeze(0)
    tt = tt.masked_fill(mask.logical_not(), 0)

    return tt.cpu(), offsets


def map_word_indexes_to_token_indexes(
    ix_words, offsets
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    enu_offsets = list(enumerate(offsets.squeeze().tolist()))
    map_ix_words_jx_tokens = {}

    pnt_tokens = 0

    for ix_word in ix_words:
        wa, wb = ix_word
        map_ix_words_jx_tokens[ix_word] = []
        joff, (a, b) = enu_offsets[pnt_tokens]
        while enu_offsets[pnt_tokens][1][1] < wb:
            map_ix_words_jx_tokens[ix_word] += [pnt_tokens]
            pnt_tokens += 1
    return map_ix_words_jx_tokens


def aggregate_token_groups(nlp, text, extra_context=False):
    doc = nlp(text)
    context_tags = ["VB", "IN", "JJ", "TO"] if extra_context else ["VB"]

    # tokens in reverse order
    tokens = [token for token in doc if token.tag_[:2] in context_tags][::-1]

    # group tokens
    acc = [[]]
    while tokens:
        ctoken = tokens.pop()
        group = acc[-1]
        if group:
            ixu, ixv = group[-1].i, ctoken.i
            if ixv - ixu == 1:
                group.append(ctoken)
            else:
                acc.append([ctoken])
        else:
            acc[0].append(ctoken)

    # remove outstanding adjectives
    def remove_bnd_adj(group):
        if group and group[0].tag_ == "JJ":
            group = group[1:]
        if group and group[-1].tag_ == "JJ":
            group = group[:-1]
        return group

    for jx, group in enumerate(acc):
        group_new = remove_bnd_adj(group)
        while len(group_new) != len(group):
            group = group_new
            group_new = remove_bnd_adj(group)
        acc[jx] = group_new

    token_groups = [
        group for group in acc if any(item.tag_[:2] == "VB" for item in group)
    ]
    return token_groups


def get_vb_spans(nlp, text, extra_context=False):
    token_groups = aggregate_token_groups(nlp, text, extra_context)
    spans = transform_tokens2spans(token_groups)
    return spans


def transform_tokens2spans(token_groups):
    spans = [(group[0].idx, group[-1].idx + len(group[-1])) for group in token_groups]
    return spans


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text)

    # split on .!? if followed by a capital and not preceded by a capital
    pat = r"(?<=[^A-Z][.!?])\s*(?=[A-Z])"
    phrases_ = re.split(pat, text)
    # trim initial/terminal whitespaces
    trim_whitespace = re.compile(r"^[\s+]+|[\s+]+$")
    phrases_ = [trim_whitespace.sub("", p) for p in phrases_]
    return phrases_


def process_text(text, tokenizer, model, nlp, max_length=None, extra_context=False):
    sents_raw = split_into_sentences(text)

    if max_length is not None:
        sents_agg = []
        for s in sents_raw:
            if sents_agg:
                if len(sents_agg[-1]) + len(s) < MAX_LENGTH - 2:
                    sents_agg[-1] = sents_agg[-1] + f" {s}"
                else:
                    sents_agg += [s]
            else:
                sents_agg += [s]
    else:
        sents_agg = sents_raw

    tt, offsets = text_to_tokens_embeddings(sents_agg, tokenizer, model)
    sent_spans = [
        sentence_ix(s, nlp, offs, extra_context=extra_context)
        for j, (offs, s) in enumerate(zip(offsets, sents_agg))
    ]
    return sents_agg, sent_spans, tt


def sentence_ix(sent, nlp, token_offsets, extra_context=False):
    spans = get_vb_spans(nlp, sent, extra_context=extra_context)

    ix_whitespaces = [0] + [
        i + 1
        for i, char in enumerate(sent)
        if char in whitespace or char in punctuation
    ]
    ix_words = list(zip(ix_whitespaces, ix_whitespaces[1:]))

    miti0 = map_word_indexes_to_token_indexes(ix_words, token_offsets)
    miti = [
        (
            (sa, sb),
            [
                it
                for (wa, wb), v in miti0.items()
                if wa >= sa and wb - 1 <= sb
                for it in v
            ],
        )
        for sa, sb in spans
    ]

    return miti


def tt_aggregate_normalize(tt: torch.Tensor, ls):
    """

    :param tt: incoming dims: n_layers x nb x n_tokens x n_emb
    :param ls:
    :return: nb x n_emb
    """

    # average over layers
    tt_norm = tt[ls].mean(0)

    n = tt_norm.norm(dim=-1)
    n[n == 0] = 1

    # normalize each tokens over embedding dim, then average over tokens
    tt_norm = (tt_norm / n.unsqueeze(-1)).mean(dim=1)

    # normalize each tokens over embedding dim, then average over tokens
    tt_norm = tt_norm / tt_norm.norm(dim=-1).unsqueeze(-1)
    return tt_norm


def compute_distance_ref(
    index,
    tt_descs,
    nb_nn,
    gt_position,
):
    distance_matrix, nearest_neighbors_matrix = index.search(tt_descs, nb_nn)
    dfd = pd.DataFrame(distance_matrix[:, 0], columns=["dist"])
    dfd["position"] = "top"
    dfd2 = pd.DataFrame(distance_matrix[:, 1:].flatten(), columns=["dist"])
    dfd2["position"] = "1-9"
    dfa = pd.concat([dfd, dfd2])
    acc_score = accuracy_score(
        nearest_neighbors_matrix[:, 0],
        gt_position,
    )

    top_mean = distance_matrix[:, 0].mean()
    below_mean = distance_matrix[:, 1:].flatten().mean()

    top_cand_dist = top_mean - below_mean
    m0 = (top_cand_dist, acc_score)
    return m0, dfa


def embedding_to_dist(tt_x, tt_y):
    index = faiss.IndexFlatIP(tt_x.shape[1])
    nb_nn = min([100, tt_x.shape[0]])
    index.add(tt_x)

    distance_matrix, nearest_neighbors_matrix = index.search(tt_y, nb_nn)
    ds = distance_matrix[:, 1:].flatten()
    ds.sort()
    k_links = 20
    thr = ds[-k_links]

    edges = []
    for dd, nn in zip(distance_matrix, nearest_neighbors_matrix):
        m = dd >= thr
        equis = nn[m]
        edges += [(equis[0], c) for c in equis[1:]]

    dfa = pd.DataFrame(ds, columns=["dist"])
    return dfa


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def encode(texts, tokenizer, model, ls):
    if ls == "sent":
        # encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        #
        # # Compute token embeddings
        # with torch.no_grad():
        #     model_output = model(**encoded_input)
        # tt_labels = mean_pooling(model_output, encoded_input['attention_mask'])

        tt_labels = model.encode(texts, normalize_embeddings=True)
    else:
        tt_labels_layered, labels_spans = text_to_tokens_embeddings(
            texts, tokenizer, model
        )
        tt_labels = tt_aggregate_normalize(tt_labels_layered, ls)
    return tt_labels


def fetch_latest_kb(path_derived) -> tuple[str | None, int]:
    file_names = [
        file.name
        for file in path_derived.iterdir()
        if file.is_file() and ".synthesis." in file.name
    ]
    filename_versions = sorted(
        [(f, int(f.split(".")[-2])) for f in file_names], key=lambda x: x[1]
    )
    if filename_versions:
        return filename_versions[-1]
    else:
        return None, -1
