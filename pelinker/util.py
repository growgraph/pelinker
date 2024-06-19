import re
from string import punctuation, whitespace

import torch


def text_to_tokens_embeddings(texts: list[str], tokenizer, model):
    encoding = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
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
    tt = tt.masked_fill((encoding["input_ids"] == 0).unsqueeze(-1), 0)

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
            # map_ix_words_jx_tokens[ix_word] += [(pnt_tokens, enu_offsets[pnt_tokens][1])]
            map_ix_words_jx_tokens[ix_word] += [pnt_tokens]
            pnt_tokens += 1
    return map_ix_words_jx_tokens


def get_vb_spans(nlp, text):
    doc = nlp(text)
    v_spans0 = [
        (token.idx, len(token.text), token.tag_[:2])
        for token in doc
        if token.tag_[:2] in ["VB", "IN", "JJ", "TO"]
        # if token.tag_[:2] in ["VB"]
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


def process_text(text, tokenizer, model, nlp):
    sents = split_into_sentences(text)
    tt, offsets = text_to_tokens_embeddings(sents, tokenizer, model)
    sent_spans = [
        sentence_ix(s, nlp, offs) for j, (offs, s) in enumerate(zip(offsets, sents))
    ]
    return sents, sent_spans, tt


def sentence_ix(sent, nlp, token_offsets):
    spans = get_vb_spans(nlp, sent)

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
