import re
from string import punctuation, whitespace
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd

from sklearn.metrics import accuracy_score

MAX_LENGTH = 512


def load_models(model_type):
    if model_type == "scibert":
        tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
        model = AutoModel.from_pretrained("allenai/scibert_scivocab_cased")
    elif model_type == "biobert":
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
    elif model_type == "pubmedbert":
        tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
    else:
        raise ValueError(f"{model_type} unsupported")
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

    inputs = {
        k: encoding[k].to(model.device)
        for k in ["input_ids", "token_type_ids", "attention_mask"]
    }

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


def get_vb_spans(nlp, text, extra_context=False):
    doc = nlp(text)
    context_tags = ["VB", "IN", "JJ", "TO"] if extra_context else ["VB"]
    v_spans0 = [
        (token.idx, len(token.text), token.tag_[:2])
        for token in doc
        if token.tag_[:2] in context_tags
    ]

    v_spans = [(a, a + s, t) for a, s, t in v_spans0]
    acc = []
    while v_spans:
        ua, ub, utag = v_spans.pop()
        if acc:
            va, vb, vtag = acc.pop()
            if va - ub == 1:
                ctag = "VB" if utag == "VB" or vtag == "VB" else vtag
                acc.append((ua, vb, ctag))
            else:
                acc += [(va, vb, vtag), (ua, ub, utag)]
        else:
            acc.append((ua, ub, utag))
    acc_vbs = sorted([x[:2] for x in acc if x[-1] == "VB"])
    return acc_vbs


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