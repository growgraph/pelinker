import click
import pandas as pd
import tqdm

from torch.nn import CosineSimilarity
from pelinker.onto import WordGrouping
from pelinker.util import texts_to_vrep
from pelinker.matching import match_pattern
import pathlib
import spacy
import torch
from pelinker.util import load_models, text_to_tokens
from pelinker.model import LinkerModel
from pelinker.util import SimplifiedToken


@click.command()
@click.option(
    "--model-type",
    type=click.STRING,
    default="biobert",
    help="run over BERT flavours",
)
@click.option(
    "--pattern",
    type=click.STRING,
    default=["dominate", "activate", "causes"],
    multiple=True,
    help="",
)
@click.option(
    "--layers-spec",
    type=click.STRING,
    default="1",
    help="`sent` or a string of layers, `1,2,3` would correspond to layers [-1, -2, -3]",
)
@click.option(
    "--input-path",
    type=click.Path(path_type=pathlib.Path),
    default="./data/test/sample.csv.gz",
    help="input df",
)
@click.option(
    "--plot-path",
    type=click.Path(path_type=pathlib.Path),
    default="figs",
    required=True,
)
def run(model_type, input_path, layers_spec, pattern, plot_path):
    nlp = spacy.load("en_core_web_trf")

    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_models(model_type, sentence=False)
    layers = LinkerModel.str2layers(layers_spec)

    df = pd.read_csv(input_path, index_col=0)
    texts = df["abstract"]
    indexes_of_interest_batch = []
    for p in pattern:
        indexes_of_interest_batch += [
            [match_pattern(p, x, suffix_length=0) for x in texts]
        ]

    pattern_expressions: dict[str, list[SimplifiedToken]] = {
        p: text_to_tokens(nlp=nlp, text=p) for p in pattern
    }
    pattern_lemmatized = [
        " ".join([e.lemma for e in pe]) for pe in pattern_expressions.values()
    ]

    flags = [
        any(True if ixs_ else False for ixs_ in item)
        for item in zip(*indexes_of_interest_batch)
    ]
    data = [t for flag, t in zip(flags, texts) if flag]

    batch_size = 40

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    frep = []
    tt_averages = []

    for ibatch, text_batch in enumerate((pbar := tqdm.tqdm(data_batched))):
        report_batch = texts_to_vrep(
            text_batch,
            tokenizer=tokenizer,
            model=model,
            layers_spec=layers,
            word_modes=[WordGrouping.W1],
            nlp=nlp,
        )

        indexes_of_interest_batch = {
            p: [match_pattern(p, text, suffix_length=0) for text in text_batch]
            for p in pattern
        }
        # offsets = [report_batch.chunk_mapper.ichunk_char_offset(k) for k, _ in enumerate(report_batch.texts)]

        for p in pattern:
            # matches = indexes_of_interest_batch[p]
            pe = pattern_expressions[p]
            pe_lemmatized = " ".join([e.lemma for e in pe])
            for w in report_batch.available_groupings():
                expression_container = report_batch[w]
                for itext, (text, expr_holder) in enumerate(
                    zip(
                        report_batch.texts,
                        expression_container.expression_data,
                    )
                ):
                    # if not matches:
                    #     continue

                    tt_averages += [expression_container.expression_data[0].tt]

                    expr_lemma_match = expr_holder.filter_on_lemmas(pe)

                    if not expr_lemma_match:
                        continue
                    offsets = [
                        report_batch.chunk_mapper.map_chunk_to_text(e.itext, e.ichunk)
                        for e, _ in expr_lemma_match
                    ]

                    frep += [
                        (
                            {
                                "word_grouping": w,
                                "idoc": ibatch * batch_size + itext,
                                "pattern_lemmatized": pe_lemmatized,
                                "mention": text[offset + e.a : offset + e.b],
                                "mention_": " ".join([t.text for t in e.tokens]),
                                "mention_lemmatized": " ".join(
                                    [t.lemma for t in e.tokens]
                                ),
                                "tensor": tt,
                            }
                        )
                        for (e, tt), offset in zip(expr_lemma_match, offsets)
                    ]

        pbar.set_description(f"entities added : {len(frep)}")

    tt_all = (torch.stack([x["tensor"] for x in frep]), "all.patterns")
    tt_pats = [
        (torch.stack([x["tensor"] for x in frep if x["pattern_lemmatized"] == p]), p)
        for p in pattern_lemmatized
    ]

    tt_means = (torch.cat(tt_averages), "all")
    cos = CosineSimilarity(dim=1, eps=1e-6)

    tts_cmp = tt_pats + [tt_all, tt_means]

    vc_mentions_combination = (
        pd.DataFrame(
            [{k: v for k, v in item.items() if k != "tensor"} for item in frep],
        )
        .apply(lambda x: ": ".join(x[["pattern_lemmatized", "mention_"]]), axis=1)
        .value_counts()
    )

    print(vc_mentions_combination)

    cos_dist = []
    for t, label in tts_cmp:
        if label == "all.patterns":
            df_labels = pd.DataFrame(
                [x["pattern_lemmatized"] for x in frep], columns=["label"]
            )
            vc_labels = (
                df_labels.groupby("label")
                .apply(lambda x: x.shape[0])
                .reset_index()
                .rename(columns={0: "cnt"})
            )
            vc_labels["weight"] = 1 / vc_labels["cnt"]
            df_labels = df_labels.merge(vc_labels, on="label")
            tt_weight = torch.from_numpy(df_labels["weight"].values)
        else:
            tt_weight = torch.ones(t.shape[:1])
        t_center = (tt_weight.unsqueeze(-1) * t).mean(0).unsqueeze(0)
        t_center_normed = (
            ((tt_weight.unsqueeze(-1) * t) / t.norm(dim=-1).unsqueeze(-1))
            .mean(0)
            .unsqueeze(0)
        )
        dists = cos(t, t_center).tolist()
        cos_dist += [(d, f"{label}") for d in dists]
        dists = cos(t, t_center_normed).tolist()
        cos_dist += [(d, f"{label}.normed") for d in dists]

    df = pd.DataFrame(cos_dist, columns=["d", "label"])
    mean_dist = df.groupby("label").apply(lambda x: x["d"].mean())

    print(df["label"].value_counts())
    print(mean_dist)

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        sns.set_style("whitegrid")
        _ = sns.displot(
            df.loc[~df["label"].apply(lambda x: x.endswith(".normed"))],
            x="d",
            hue="label",
            kind="kde",
            common_norm=False,
            # stat="density",
            # element="step",
            # bins=20
        )

        plt.savefig(
            plot_path.expanduser() / f"cos_sim_dist.{model_type}.{layers_spec}.pdf",
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print(f"something happened : {e}")


if __name__ == "__main__":
    run()
