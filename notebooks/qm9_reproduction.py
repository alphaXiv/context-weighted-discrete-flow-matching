# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.14.17",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Context-weighted discrete flow matching on QM9

    A molecule can be written as a sequence of symbols, but a locally
    inconsistent reveal can make the whole sequence chemically invalid.
    The paper proposes revealing well-contextualized positions faster and
    weighting their training loss more heavily. This notebook opens with
    the already-computed evidence; it does **not** rerun expensive training.

    **Verdict: partially reproduced.** Neighbor weighting improves validity
    and novelty after enough sampling updates, and local context strongly
    predicts uncertainty. It regresses at very low NFE, while scaled
    cross-entropy does not improve this compact model.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ![Primary result](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/sampler_by_nfe.png)

    Each point averages four independently trained models; error bars are
    seed standard deviations. Neighbor weighting crosses Euler between
    NFE 32 and 64.
    """)
    return


@app.cell
def _():
    nfe = [8, 16, 32, 64, 128, 256]
    sampler_results = {
        "Euler": {
            "valid": [237.2, 347.0, 412.0, 443.5, 451.5, 480.2],
            "novel": [210.0, 291.0, 337.8, 360.0, 373.5, 390.5],
        },
        "Neighbor": {
            "valid": [210.8, 332.0, 416.2, 467.0, 483.5, 503.2],
            "novel": [183.2, 285.0, 347.5, 383.8, 391.0, 406.0],
        },
        "Entropy": {
            "valid": [51.5, 130.8, 269.0, 379.2, 470.2, 542.5],
            "novel": [48.0, 121.8, 236.0, 319.8, 383.8, 424.8],
        },
    }
    return nfe, sampler_results


@app.cell
def _(mo, nfe):
    nfe_picker = mo.ui.dropdown(
        options={str(value): value for value in nfe},
        value="128",
        label="Inspect NFE",
    )
    nfe_picker
    return (nfe_picker,)


@app.cell
def _(mo, nfe, nfe_picker, sampler_results):
    index = nfe.index(nfe_picker.value)
    table = [
        {
            "sampler": method,
            "valid / 1,024": values["valid"][index],
            "novel / 1,024": values["novel"][index],
        }
        for method, values in sampler_results.items()
    ]
    mo.vstack(
        [
            mo.md(f"### Matched-compute comparison at NFE {nfe_picker.value}"),
            mo.ui.table(table, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Mechanism: nearby reveals make predictions easier

    ![Context mechanism](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/context_mechanism.png)

    At path time 0.5, mean predictive entropy falls monotonically from
    **1.05 nats** with zero revealed neighbors to **0.29 nats** with four.
    Negative log-likelihood follows the same curve. This directly supports
    the paper's use of local context as an observable uncertainty proxy.
    """)
    return


@app.cell
def _():
    def normalized_neighbor_weights(visible_neighbors, inverse_temperature=4.0):
        """The inference intervention in its smallest conceptual form."""
        import math

        scores = [math.exp(inverse_temperature * count) for count in visible_neighbors]
        total = sum(scores)
        return [score / total for score in scores]

    example_weights = normalized_neighbor_weights([0, 1, 2])
    return (example_weights,)


@app.cell
def _(example_weights, mo):
    mo.md(
        f"""
        For masked positions with 0, 1, and 2 visible neighbors, the normalized
        priority weights at scale 4 are approximately
        `{[round(value, 4) for value in example_weights]}`. The sampler changes
        *where* the next reveals occur while keeping the number of reveals
        matched to Euler.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Training intervention and radius

    ![Radius ablation](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/radius_ablation.png)

    The paper's masked-source table improves from 134.8 to 177.2 valid
    molecules with scaled cross-entropy. Here CE reaches **480.2±18.3**
    valid molecules at NFE 256, while the best SCE radius reaches
    **419.2±8.3**. Radius 1 is best among SCE variants, but no radius
    recovers the CE baseline.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Temperature robustness

    ![Temperature robustness](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/temperature_robustness.png)

    Lower temperature improves absolute validity, but the high-NFE
    neighbor-over-Euler ordering persists at 0.8, 1.0, and 1.2. Neighbor
    scales 2, 4, and 6 also produce similar seed-0 NFE-256 validity
    (480, 493, and 492).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What exactly ran

    - Public DeepChem QM9 SDF, canonicalized with RDKit; 110k training molecules.
    - Masked-source, length-32, 14.4M-parameter bidirectional transformer.
    - 25k AdamW steps; four seeds for CE and SCE radii 1/2/3.
    - Euler, neighbor, and entropy sampling at NFE 8–256; 1,024 generations per condition.
    - Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs; peak 16 concurrent GPUs.
    - Actual launch-to-last-result wall time: 0.51 hours.

    The architecture (14.4M vs 92M parameters), batch size (512 vs 2,048),
    and split are substitutions. This QM9 reproduction does not test the
    paper's OpenWebText claims or context-weighted train-time path.

    [Detailed report](https://github.com/alphaXiv/context-weighted-discrete-flow-matching/blob/main/reports/qm9/report.md)
    · [Factorial CSV](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/data/factorial_results.csv)
    · [Robustness CSV](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/data/robustness_results.csv)
    """)
    return


if __name__ == "__main__":
    app.run()
