# pylint: disable=E1120

import re

from string import punctuation, whitespace
from typing import List

import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pandas as pd
from sklearn.metrics import accuracy_score
from functools import reduce
import logging

from pelinker.onto import (
    MAX_LENGTH,
    ChunkMapper,
    WordGrouping,
    SimplifiedToken,
    Expression,
    ExpressionHolder,
    ExpressionHolderBatch,
    ReportBatch,
    _wg_for_property,
)

logger = logging.getLogger(__name__)


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
    """

    :param texts:
    :param tokenizer:
    :param model:
    :return: tensor of encoded tokes, token boundaries
    """

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

    # n_layers x n_batch x n_len x n_emb
    tt = torch.stack(outputs.hidden_states)

    # fill with zeros latent vectors for padded tokens
    mask = encoding["attention_mask"]
    mask = mask.unsqueeze(-1).unsqueeze(0)
    tt = tt.masked_fill(mask.logical_not(), 0)

    # nb x n_len x 2
    offsets = encoding["offset_mapping"]
    enu_offsets = [
        [(x, y) for x, y in sublist if x != y] for sublist in offsets.tolist()
    ]

    return tt.cpu(), enu_offsets


def map_spans_to_spans_basic(
    words_boundaries, token_boundaries
) -> dict[tuple[int, int], list[int]]:
    """
        given two lists of word bounds and token bounds (in character indexes)
        [ it is implied that the two lists are sorted ]
        the list of token boundaries is meant to cover the text (to be complete),
        on the other hand word boundaries might be not `continuous`

        find the correspondence: (wa, wb) -> [index of token]

        to each word boundary find the indexes of corresponding tokens

    :param words_boundaries:
    :param token_boundaries:
    :return: dict with
                key: (char_a, char_b) boundary of group of interest (words)
                value: list of corresponding tokens
    """

    map_ix_jx = {}

    pnt_tokens = 0

    for ix_word in words_boundaries:
        wa, wb = ix_word
        map_ix_jx[ix_word] = []
        while (
            pnt_tokens < len(token_boundaries) and token_boundaries[pnt_tokens][1] <= wb
        ):
            if token_boundaries[pnt_tokens][0] >= wa:
                map_ix_jx[ix_word] += [pnt_tokens]
            pnt_tokens += 1
    return map_ix_jx


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


def get_vb_spans(nlp, text, extra_context=False) -> list[tuple[int, int]]:
    token_groups = aggregate_token_groups(nlp, text, extra_context)
    spans = transform_tokens2spans(token_groups)
    return spans


def text_to_tokens(nlp, text) -> list[SimplifiedToken]:
    stokens = [
        SimplifiedToken(
            **{
                "lemma": token.lemma_,
                "text": token.text,
                "tag": token.tag_,
                "ix": token.idx,
                "ix_end": token.idx + len(token),
            }
        )
        for token in nlp(text)
    ]

    return stokens


def transform_tokens2spans(token_groups) -> list[tuple[int, int]]:
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


def process_text(batched_texts: list[list[str]], tokenizer, model) -> ChunkMapper:
    """

    :param batched_texts:
    :param tokenizer:
    :param model:
    :return: ChunkMapper
    """

    flattened_chunks: list[str] = [s for group in batched_texts for s in group]

    it_ic = [
        (i_t, i_c) for i_t, sl in enumerate(batched_texts) for i_c, _ in enumerate(sl)
    ]

    chunk_cumlens = [
        reduce(lambda a, x: a + [a[-1] + x], [len(item) for item in chunks], [0])
        for chunks in batched_texts
    ]

    tt, offsets = text_to_tokens_embeddings(flattened_chunks, tokenizer, model)

    return ChunkMapper(
        tensor=tt,
        chunks=flattened_chunks,
        token_spans_list=offsets,
        it_ic=it_ic,
        cumulative_lens=chunk_cumlens,
    )


def map_words_to_tokens(
    text_token_spans: list[tuple[int, int]], text_word_spans: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    given text token and word spans,

        words : [...(start_pos_i, end_pos_i)...]
        tokens : [...(start_pos_i, end_pos_i)...]

    we define work -> token spans, i.e. a mapping of which tokens belong to words groups

    if there is a positive overlap between word i and token j spans,
        we consider that token j belongs to work i group

    return a list of work -> token bounds : [...(start_token_i, end_token_i)...]
    and a refreshed


    """

    map_ix_jx = map_spans_to_spans_basic(text_word_spans, text_token_spans)

    # sanitize token offsets
    map_ix_jx = {k: v for k, v in map_ix_jx.items() if v}
    text_word_spans = [x for x in map_ix_jx.keys()]
    token_word_spans = [(y[0], y[-1] + 1) for y in map_ix_jx.values()]
    return text_word_spans, token_word_spans


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


def tt_normalize(cm: ChunkMapper, layers) -> list[list[torch.Tensor]]:
    """

    :param cm:
    :param layers:
    :return:
    """
    tt_norm = cm.tensor[layers].mean(0)
    tt_r = []
    for js, (ix_tokens, tt_sent) in enumerate(zip(cm.token_word_spans_list, tt_norm)):
        tt_words_list = []
        for a, b in ix_tokens:
            rr = tt_sent[a:b].mean(0)
            # rr = rr / rr.norm(dim=-1).unsqueeze(-1)
            tt_words_list += [rr]
        tt_r += [tt_words_list]
    return tt_r


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


def split_long_text(text, max_length=MAX_LENGTH):
    """

    :param text:
    :param max_length:
    :return: list of strings representing text, such that each string is < max_length
        text = " ".join(agg)
    """
    agg = []

    for chunk in split_text_into_batches(text, max_length - 2):
        if agg:
            if len(agg[-1]) + len(chunk) < max_length - 2:
                agg[-1] = agg[-1] + f" {chunk}"
            else:
                agg += [chunk]
        else:
            agg += [chunk]

    return agg


def split_text_into_batches(text: str, max_length) -> list[str]:
    pattern = (
        r"(.{1," + str(max_length - 1) + r"})(\s|$)|(.{1," + str(max_length) + r"})"
    )

    matches = re.findall(pattern, text)
    batched = ["".join(parts) for parts in matches]
    return batched


def get_word_boundaries(text) -> list[tuple[int, int]]:
    """
        render word boundaries in text

    :param text:
    :return:
    """

    ix_whitespaces = [0] + [
        i + 1
        for i, char in enumerate(text)
        if char in whitespace or char in punctuation
    ]
    if text[-1] not in whitespace and text[-1] not in punctuation:
        ix_whitespaces += [len(text) + 1]

    ix_words = [(i, j - 1) for i, j in zip(ix_whitespaces, ix_whitespaces[1:])]
    ix_words = [(i, j) for i, j in ix_words if i != j]
    return ix_words


def render_elementary_tensor_table(
    chunk_mapper: ChunkMapper, text_word_spans: list[list[tuple[int, int]]], layers
) -> None:
    """

    Args:
        chunk_mapper:
        text_word_spans:
        layers:

    Returns:

    """

    chunk_mapper.set_token_word_spans(text_word_spans)
    chunk_mapper.set_mapping_table()

    ll_tt = tt_normalize(chunk_mapper, layers)
    chunk_mapper.tt_expressions = [
        torch.stack([t for t in sl]) if sl else torch.tensor([]) for sl in ll_tt
    ]


def build_expression_container(
    cm: ChunkMapper, expression_lists_per_chunk, word_grouping: WordGrouping
) -> ExpressionHolderBatch:
    texts = []
    for itext, ichunks in cm.text_chunk_map.items():
        expressions = reduce(
            lambda a, b: a + b, [expression_lists_per_chunk[i] for i in ichunks]
        )
        tt_merged = torch.cat([cm.tt_expressions[i] for i in ichunks])
        texts.append(ExpressionHolder(tt=tt_merged, expressions=expressions))

    return ExpressionHolderBatch(expression_data=texts, word_grouping=word_grouping)


def texts_to_vrep(
    texts: list[str],
    tokenizer,
    model,
    layers_spec,
    word_modes: list[WordGrouping],
    max_length=MAX_LENGTH,
    nlp=None,
) -> ReportBatch:
    batched_texts: list[list[str]] = [
        split_text_into_batches(s, max_length=max_length) for s in texts
    ]

    chunk_mapper: ChunkMapper = process_text(
        batched_texts,
        tokenizer,
        model,
    )

    stoken_per_chunk: list[list[SimplifiedToken]] = [
        text_to_tokens(nlp=nlp, text=chunk) for chunk in chunk_mapper.chunks
    ]

    # ichunk -> itext, ichunk local
    ichunk_to_itext_ichunk_local_list = [
        chunk_mapper.ichunk_to_itext_ichunk_local(k)
        for k, _ in enumerate(stoken_per_chunk)
    ]

    data: list[ExpressionHolderBatch] = []
    for word_grouping in word_modes:
        expression_lists_per_chunk: list[list[Expression]] = [
            token_list_with_window(
                chunk_tokens, word_grouping, ichunk=ichunk_local, itext=itext
            )
            for chunk_tokens, (itext, ichunk_local) in zip(
                stoken_per_chunk, ichunk_to_itext_ichunk_local_list
            )
        ]

        word_spans: list[list[tuple[int, int]]] = [
            [(e.a, e.b) for e in batch] for batch in expression_lists_per_chunk
        ]

        render_elementary_tensor_table(chunk_mapper, word_spans, layers_spec)

        # adjust expressions
        filtered_expression_lists_per_chunk: list[list[Expression]] = []
        for exprs, word_spans in zip(
            expression_lists_per_chunk, chunk_mapper.text_word_spans_list
        ):
            ix_start = {a for a, _ in word_spans}
            filtered_expression_lists_per_chunk.append(
                [e for e in exprs if e.a in ix_start]
            )

        data += [
            build_expression_container(
                chunk_mapper,
                filtered_expression_lists_per_chunk,
                word_grouping=word_grouping,
            )
        ]
    return ReportBatch(chunk_mapper=chunk_mapper, texts=texts, _data=data)


def report2kb(report, wg):
    wg_current = report["word_groupings"][wg]
    tt_list = []
    vocabulary = []
    for sentence in wg_current:
        tt_list += [t for _, t in sentence]
        vocabulary += [item["mention"] for item, _ in sentence]
    tt_basis = torch.concat(tt_list)
    index = faiss.IndexFlatIP(tt_basis.shape[1])
    index.add(tt_basis)
    return vocabulary, index


def merge_wbs(word_boundaries, window) -> list[tuple[int, int]]:
    return [
        (wa, wb)
        for (wa, _), (_, wb) in zip(word_boundaries, word_boundaries[window - 1 :])
    ]


def token_list_with_window(
    tokens: list[SimplifiedToken], window: WordGrouping, itext=None, ichunk=None
) -> list[Expression]:
    agg = []
    w = int(window.value)
    for k in range(len(tokens) - w + 1):
        agg.append(Expression(tokens=tokens[k : k + w], itext=itext, ichunk=ichunk))
    return agg


def map_words_to_tokens_list(
    text_token_spans_list: list[list[tuple[int, int]]],
    text_word_spans_list: list[list[tuple[int, int]]],
):
    """
    take a batch of token spans and a batch of word spans

    return a batch of work to token maps
        and also an updated batch of word spans
            (some words can not be mapped to tokens, so they are excluded)


    """

    text_word_spans_list_: list[list[tuple[int, int]]] = []
    token_work_spans_list: list[list[tuple[int, int]]] = []

    for text_token_spans, text_word_spans in zip(
        text_token_spans_list, text_word_spans_list
    ):
        text_word_spans, token_word_spans = map_words_to_tokens(
            text_token_spans, text_word_spans
        )
        text_word_spans_list_ += [text_word_spans]
        token_work_spans_list += [token_word_spans]

    return token_work_spans_list, text_word_spans_list_


def extract_and_embed_mentions(
    props: list[str],
    data: list[str],
    pmids: list[str],
    tokenizer,
    model,
    nlp,
    layers,
    batch_size,
    word_modes=(WordGrouping.W1, WordGrouping.W2, WordGrouping.W3),
) -> List[dict]:
    """
    Modified to return list of dicts instead of DataFrame for better memory management
    and consistent schema handling.
    """
    data_pmids = pmids

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    data_pmids_batched = [
        data_pmids[i : i + batch_size] for i in range(0, len(data), batch_size)
    ]

    # Pre-tokenize properties for lemma matching
    prop_tokens = {p: text_to_tokens(nlp=nlp, text=p) for p in props}

    rows = []
    for ibatch, text_batch in enumerate((pbar := tqdm.tqdm(data_batched))):
        report_batch = texts_to_vrep(
            text_batch,
            tokenizer=tokenizer,
            model=model,
            layers_spec=layers,
            word_modes=list(word_modes),
            nlp=nlp,
        )

        batch_pmids = data_pmids_batched[ibatch]

        # For each property, pick the matching word grouping and aggregate matches
        for p in props:
            pe = prop_tokens[p]
            wg = _wg_for_property(p)
            if wg is None:
                continue
            if wg not in report_batch.available_groupings():
                continue

            expression_container = report_batch[wg]
            for itext, (text, expr_holder) in enumerate(
                zip(report_batch.texts, expression_container.expression_data)
            ):
                expr_lemma_match = expr_holder.filter_on_lemmas(pe)
                if not expr_lemma_match:
                    continue

                offsets = [
                    report_batch.chunk_mapper.map_chunk_to_text(e.itext, e.ichunk)
                    for e, _ in expr_lemma_match
                ]

                for (e, tt), offset in zip(expr_lemma_match, offsets):
                    mention = text[offset + e.a : offset + e.b]
                    # Convert numpy array to list for consistent Parquet schema
                    embed_list = tt.numpy().tolist()
                    rows.append(
                        {
                            "pmid": batch_pmids[itext],
                            "property": p,
                            "mention": mention,
                            "embed": embed_list,  # Now a Python list, not numpy array
                        }
                    )

        pbar.set_description(f"Entities added in chunk : {len(rows)}")

    return rows
