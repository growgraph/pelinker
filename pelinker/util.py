# pylint: disable=E1120

import re

from spacy import Language
from string import punctuation, whitespace
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
    ExpressionBatch,
    ExpressionContainer,
)
from collections import defaultdict

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
        (i_c, i_t) for i_t, sl in enumerate(batched_texts) for i_c, _ in enumerate(sl)
    ]

    chunk_cumlens = [
        reduce(lambda a, x: a + [a[-1] + x], [len(item) for item in chunks], [0])
        for chunks in batched_texts
    ]

    tt, offsets = text_to_tokens_embeddings(flattened_chunks, tokenizer, model)

    return ChunkMapper(
        tt=tt,
        flattened_chunks=flattened_chunks,
        token_bounds=offsets,
        it_ic=it_ic,
        chunk_cumlens=chunk_cumlens,
    )


def compute_spans(
    token_bnds: list[list[tuple[int, int]]], word_bnds: list[list[tuple[int, int]]]
):
    sent_spans = [
        map_spans_to_spans(t_bnds, w_bnds)
        for t_bnds, w_bnds in zip(token_bnds, word_bnds)
    ]

    char_spans: list[list[tuple[int, int]]] = [x for x, _ in sent_spans]
    token_spans: list[list[tuple[int, int]]] = [x for _, x in sent_spans]
    return token_spans, char_spans


def sentence_ix(
    sent: str,
    token_offsets: torch.tensor,  # 2D
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ix_words = get_word_boundaries(sent)

    map_ix_jx = map_spans_to_spans_basic(ix_words, token_offsets)

    # sanitize
    map_ix_jx = {k: v for k, v in map_ix_jx.items() if v}
    char_spans = [x for x in map_ix_jx.keys()]
    itoken_spans = [(y[0], y[-1] + 1) for y in map_ix_jx.values()]
    return char_spans, itoken_spans


def map_spans_to_spans(
    token_bounds: list[tuple[int, int]], word_bounds: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    map_ix_jx = map_spans_to_spans_basic(word_bounds, token_bounds)

    # sanitize token offsets
    map_ix_jx = {k: v for k, v in map_ix_jx.items() if v}
    # if len(map_ix_jx) != len(map_ix_jx_):
    #     logger.info(f"{map_ix_jx_} {map_ix_jx}")
    char_spans = [x for x in map_ix_jx.keys()]
    itoken_spans = [(y[0], y[-1] + 1) for y in map_ix_jx.values()]
    return char_spans, itoken_spans


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


def tt_normalize(
    tt: torch.Tensor, ls, ix_tokens: list[list[tuple[int, int]]]
) -> list[list[torch.Tensor]]:
    """

    :param tt: n_layers x n_chunks x n_tokens x emb_dim
    :param ls:
    :param ix_tokens: list of size n_chunks
    :return:
    """
    tt_norm = tt[ls].mean(0)
    tt_r = []
    for js, (ix_tokens, tt_sent) in enumerate(zip(ix_tokens, tt_norm)):
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


def render_tensor_per_group(chunk_mapper: ChunkMapper, layers, token_spans):
    mapping_table = []
    for (ichunk, (ichunk_local, itext)), chsp in zip(
        enumerate(chunk_mapper.it_ic), chunk_mapper.char_spans
    ):
        chunk_offset = chunk_mapper.chunk_cumlens[itext][ichunk_local]
        for a, b in chsp:
            mapping_table += [
                (itext, ichunk, (a, b), (a + chunk_offset, b + chunk_offset))
            ]

    ll_tt = tt_normalize(chunk_mapper.tt, layers, chunk_mapper.token_spans)
    ll_tt_stacked = torch.stack([t for sl in ll_tt for t in sl])
    return ll_tt_stacked, mapping_table


def render_elementary_tensor_table(
    chunk_mapper: ChunkMapper, word_int_bounds: list[list[tuple[int, int]]], layers
) -> tuple[torch.Tensor, list[tuple]]:
    token_spans, char_spans = compute_spans(chunk_mapper.token_bounds, word_int_bounds)
    chunk_mapper.token_spans = token_spans
    chunk_mapper.char_spans = char_spans

    mapping_table = []
    for (ichunk, (ichunk_local, itext)), chsp in zip(
        enumerate(chunk_mapper.it_ic), chunk_mapper.char_spans
    ):
        chunk_offset = chunk_mapper.chunk_cumlens[itext][ichunk_local]
        for a, b in chsp:
            mapping_table += [
                (itext, ichunk, (a, b), (a + chunk_offset, b + chunk_offset))
            ]

    ll_tt = tt_normalize(chunk_mapper.tt, layers, chunk_mapper.token_spans)
    ll_tt_stacked = torch.stack([t for sl in ll_tt for t in sl])
    return ll_tt_stacked, mapping_table

    # for itext, ichunk, (_a, _b), (a, b) in mapping_table:
    #     print(texts[itext][a:b], chunk_mapper.flattened_chunks[ichunk][_a:_b])
    #     assert texts[itext][a:b] == chunk_mapper.flattened_chunks[ichunk][_a:_b]


def derive_bounds_to_tensors(
    chunk_mapper: ChunkMapper, word_bnds, layers_spec, texts: list[str]
) -> tuple[list, torch.Tensor]:
    """

    :param chunk_mapper:
    :param word_bnds:
    :param layers_spec:
    :param texts:
    :return:
    """

    ll_tt_stacked, mapping_table = render_elementary_tensor_table(
        chunk_mapper, word_bnds, layers_spec
    )

    report = []
    for row in mapping_table:
        itext, ichunk, _, (span_a, span_b) = row
        report += [
            {
                "itext": itext,
                "a": span_a,
                "b": span_b,
                "mention": texts[itext][span_a:span_b],
            }
        ]

    return report, ll_tt_stacked


def texts_to_stokens(
    batched_texts: list[list[str]], nlp: Language
) -> list[list[SimplifiedToken]]:
    """
        take a list of texts and provide embeddings based on `word_mode`

    :param batched_texts: list of strings
    :param nlp:
    :return:
    """
    batched_stokens: list[list[SimplifiedToken]] = [
        text_to_tokens(
            nlp=nlp,
            text=s,
        )
        for batch in batched_texts
        for s in batch
    ]

    return batched_stokens


def align_expressions_to_mapping(
    expression_llist: list[list[Expression]], mapping_table: list[tuple]
):
    expression_list = [
        Expression.from_dict({**x.to_dict(), **{"ibatch": ibatch}})
        for ibatch, sl in enumerate(expression_llist)
        for x in sl
    ]
    expression_list_filtered = []

    ix = 0
    gap = 0
    while ix < len(mapping_table):
        e = expression_list[ix + gap]
        m = mapping_table[ix]
        isent, ibatch, bnds, abs_bnds = m

        if e.ibatch == ibatch and e.a == bnds[0] and e.b == bnds[1]:
            e.itext = isent
            expression_list_filtered.append(e)
            ix += 1
        else:
            gap += 1

    if len(expression_list_filtered) != len(mapping_table):
        raise ValueError(
            "expression_list_filtered should contain all entries from mapping_table"
        )

    return expression_list_filtered


def build_expression_container(
    expressions: list[Expression], tt
) -> ExpressionContainer:
    grouped = defaultdict(list)

    for idx, expr in enumerate(expressions):
        grouped[(expr.ibatch, expr.itext)].append((idx, expr))

    batches = []
    for _, items in grouped.items():
        exprs = [e for _, e in items]

        start_idx = items[0][0]
        end_idx = items[-1][0] + 1

        idx_tensor = tt[start_idx:end_idx, ...]

        batches.append(ExpressionBatch(tt=idx_tensor, expressions=exprs))

    return ExpressionContainer(batches=batches)


def texts_to_vrep(
    texts: list[str],
    tokenizer,
    model,
    layers_spec,
    word_modes: list[WordGrouping],
    max_length=MAX_LENGTH,
    nlp=None,
):
    batched_texts: list[list[str]] = [
        split_text_into_batches(s, max_length=max_length) for s in texts
    ]

    chunk_mapper: ChunkMapper = process_text(
        batched_texts,
        tokenizer,
        model,
    )

    normalized_text = ["".join(chunk) for chunk in batched_texts]

    report0: dict[WordGrouping, list[list[tuple[dict, torch.Tensor]]]] = {}

    batched_stokens: list[list[SimplifiedToken]] = [
        text_to_tokens(
            nlp=nlp,
            text=s,
        )
        for batch in batched_texts
        for s in batch
    ]

    for window in word_modes:
        expression_llist = [
            token_list_with_window(tokens, window)
            for ix, tokens in enumerate(batched_stokens)
        ]
        word_bnds = [[(e.a, e.b) for e in batch] for batch in expression_llist]

        ll_tt_stacked, mapping_table = render_elementary_tensor_table(
            chunk_mapper, word_bnds, layers_spec
        )
        print(ll_tt_stacked.shape)

        report, ll_tt_stacked = derive_bounds_to_tensors(
            chunk_mapper, word_bnds, layers_spec, normalized_text
        )

        expressions = align_expressions_to_mapping(
            expression_llist=expression_llist, mapping_table=mapping_table
        )
        report0[window]: ExpressionContainer = build_expression_container(
            expressions=expressions, tt=ll_tt_stacked
        )

    return {"normalized_text": texts, "word_groupings": report0}


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
    tokens: list[SimplifiedToken], window: WordGrouping
) -> list[Expression]:
    agg = []
    w = int(window.value)
    for k in range(len(tokens) - w + 1):
        agg.append(Expression(tokens=tokens[k : k + w]))
    return agg
