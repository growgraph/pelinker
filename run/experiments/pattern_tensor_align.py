import click
import pandas as pd
import tqdm

from torch.nn import CosineSimilarity
from pelinker.onto import WordGrouping
from pelinker.util import texts_to_vrep
from pelinker.matching import match_pattern
import pathlib
import torch
from pelinker.util import load_models
from pelinker.model import LinkerModel
from pelinker.util import map_spans_to_spans


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
    if not plot_path.exists():
        plot_path.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_models(model_type, sentence=False)
    layers = LinkerModel.str2layers(layers_spec)

    df = pd.read_csv(input_path, index_col=0)
    texts = df["abstract"]
    indexes_of_interest_per_pat = []
    for p in pattern:
        indexes_of_interest_per_pat += [
            [match_pattern(p, x, suffix_length=0) for x in texts]
        ]

    flags = [
        any(True if ixs_ else False for ixs_ in item)
        for item in zip(*indexes_of_interest_per_pat)
    ]
    data = [t for flag, t in zip(flags, texts) if flag]

    batch_size = 40

    data_batched = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    frep = []
    mean_tts = []
    for batch in (pbar := tqdm.tqdm(data_batched)):
        report = texts_to_vrep(
            batch,
            tokenizer=tokenizer,
            model=model,
            layers_spec=layers,
            word_modes=[WordGrouping.W1],
        )

        for p in pattern:
            normalized_texts = report["normalized_text"]
            word_groupings = report["word_groupings"]
            for w, r_item in word_groupings.items():
                for jsent, (text, report_sent) in enumerate(
                    zip(normalized_texts, r_item)
                ):
                    if p == pattern[0] and w == sorted(word_groupings)[0]:
                        tt0 = torch.stack([t for _, t in report_sent]).mean(0)
                        mean_tts += [tt0]

                    indexes_of_interest_batched = match_pattern(
                        p, text, suffix_length=0
                    )
                    if not indexes_of_interest_batched:
                        continue
                    report_sent = sorted(report_sent, key=lambda x: x[0]["a"])

                    _, map_ij = map_spans_to_spans(
                        [(x["a"], x["b"]) for x, _ in report_sent],
                        indexes_of_interest_batched,
                    )
                    tts = [
                        torch.stack([t for _, t in report_sent[ja:jb]]).mean(0)
                        for ja, jb in map_ij
                    ]

                    mentions = [
                        " ".join([x["mention"] for x, _ in report_sent[ja:jb]])
                        for (ja, jb) in map_ij
                    ]
                    frep += [(w, jsent, p, m, tt) for m, tt in zip(mentions, tts)]
        pbar.set_description(f"entities added : {len(frep)}")

    tt_all = (torch.stack([x[4] for x in frep]), "all.patterns")
    tt_pats = [(torch.stack([x[4] for x in frep if x[2] == p]), p) for p in pattern]

    tt_means = (torch.stack(mean_tts), "means")
    cos = CosineSimilarity(dim=1, eps=1e-6)

    tts_cmp = tt_pats + [tt_all, tt_means]

    vc_mentions_combication = (
        pd.DataFrame(
            [item[:4] for item in frep], columns=["wg", "isent", "pat", "mention"]
        )
        .apply(lambda x: ": ".join(x[["pat", "mention"]]), axis=1)
        .value_counts()
    )

    print(vc_mentions_combication)

    cos_dist = []
    for t, label in tts_cmp:
        tt_all_center = t.mean(0).unsqueeze(0)
        tt_all_center_normed = (t / t.norm(dim=-1).unsqueeze(-1)).mean(0).unsqueeze(0)
        dists = cos(t, tt_all_center).tolist()
        cos_dist += [(d, f"{label}") for d in dists]
        dists = cos(t, tt_all_center_normed).tolist()
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
