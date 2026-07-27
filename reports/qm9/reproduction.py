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

    Molecular generators reveal a string one token at a time. The paper argues
    that a hidden token with visible neighbors is less uncertain, so revealing
    locally supported tokens first should yield better molecules. We trained
    compact masked Transformers on public QM9 SMILES to test that mechanism and
    its proposed sampling and training interventions.

    ## Verdict: not reproduced in this compact setup

    Local context strongly predicted lower uncertainty, but neighbor weighting
    hurt validity and novelty at the lowest model-evaluation counts. A
    reconstructed scaled cross-entropy objective also underperformed ordinary
    cross-entropy at every tested radius.

    ![Four-seed QM9 sampling result](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/sampling_nfe.png)

    The figure is already-produced Kubernetes evidence, not a notebook rerun.
    Points are means over four independently trained models and bands show one
    sample standard deviation.
    """)
    return


@app.cell
def _():
    sampling_summary = [
        {"NFE": 8, "Euler valid": 237.2, "Neighbor valid": 210.8, "Euler novel": 210.0, "Neighbor novel": 183.2},
        {"NFE": 16, "Euler valid": 347.0, "Neighbor valid": 332.0, "Euler novel": 291.0, "Neighbor novel": 285.0},
        {"NFE": 32, "Euler valid": 412.0, "Neighbor valid": 416.2, "Euler novel": 337.8, "Neighbor novel": 347.5},
        {"NFE": 64, "Euler valid": 443.5, "Neighbor valid": 467.0, "Euler novel": 360.0, "Neighbor novel": 383.8},
        {"NFE": 128, "Euler valid": 451.5, "Neighbor valid": 483.5, "Euler novel": 373.5, "Neighbor novel": 391.0},
        {"NFE": 256, "Euler valid": 480.2, "Neighbor valid": 503.2, "Euler novel": 390.5, "Neighbor novel": 406.0},
    ]
    return (sampling_summary,)


@app.cell
def _(mo):
    nfe_picker = mo.ui.slider(
        start=8,
        stop=256,
        step=8,
        value=8,
        label="Model evaluations (choose a listed power of two)",
    )
    nfe_picker
    return (nfe_picker,)


@app.cell
def _(mo, nfe_picker, sampling_summary):
    listed_nfes = [row["NFE"] for row in sampling_summary]
    nearest_nfe = min(listed_nfes, key=lambda value: abs(value - nfe_picker.value))
    selected_sampling = next(
        row for row in sampling_summary if row["NFE"] == nearest_nfe
    )
    validity_ratio = (
        selected_sampling["Neighbor valid"] / selected_sampling["Euler valid"]
    )
    novelty_ratio = (
        selected_sampling["Neighbor novel"] / selected_sampling["Euler novel"]
    )
    mo.vstack(
        [
            mo.md(
                f"""
                **Nearest measured NFE: {nearest_nfe}.**
                Neighbor/Euler = **{validity_ratio:.3f}× validity** and
                **{novelty_ratio:.3f}× novelty**.
                """
            ),
            mo.ui.table([selected_sampling]),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The mechanism does appear

    At the midpoint of the masking path, adding visible radius-two neighbors
    monotonically reduced both predictive entropy and error. This is the
    cleanest aligned result: the model really did know more when local context
    was available.

    ![Uncertainty diagnostic](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/context_uncertainty.png)
    """)
    return


@app.cell
def _():
    mechanism_summary = [
        {"visible neighbors": 0, "entropy": 1.062, "token NLL": 1.044},
        {"visible neighbors": 1, "entropy": 0.822, "token NLL": 0.800},
        {"visible neighbors": 2, "entropy": 0.604, "token NLL": 0.574},
        {"visible neighbors": 3, "entropy": 0.434, "token NLL": 0.415},
        {"visible neighbors": 4, "entropy": 0.283, "token NLL": 0.267},
    ]
    return (mechanism_summary,)


@app.cell
def _(mechanism_summary, mo):
    mo.ui.table(mechanism_summary)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. How the matched-compute samplers differ

    Euler independently flips masked positions using the schedule's update
    probability. The weighted samplers first draw the same update count, then
    allocate those updates by local evidence. Thus all methods make the same
    number of Transformer calls; only the position selection changes.

    ```python
    count = Binomial(number_masked, update_probability).sample()
    local = visible_neighbor_count(masked_sequence, radius=1)
    scores = 4.0 * local                       # neighbor sampler
    # scores = -7.0 * predictive_entropy      # entropy sampler
    chosen = multinomial(softmax(scores), count, replacement=False)
    ```

    Temperature 0.8–1.2 and neighbor-scale checks did not rescue the low-NFE
    effect: every single-seed setting lowered validity at 8 and 16 evaluations.
    Four-seed temperature repeats agreed at NFE 8; temperature 1.2 was neutral
    at NFE 16. The checks did show a repeatable modest gain at 64–256 evaluations.

    ![Sampling robustness](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/robustness.png)
    """)
    return


@app.cell
def _():
    loss_summary = [
        {"objective": "ordinary CE", "radius": "—", "valid": "480.3 ± 18.3", "novel": "390.5 ± 6.4"},
        {"objective": "scaled CE", "radius": 1, "valid": "419.3 ± 8.3", "novel": "345.8 ± 10.2"},
        {"objective": "scaled CE", "radius": 2, "valid": "393.0 ± 12.2", "novel": "329.0 ± 10.9"},
        {"objective": "scaled CE", "radius": 3, "valid": "402.5 ± 7.1", "novel": "336.3 ± 4.7"},
    ]
    return (loss_summary,)


@app.cell
def _(loss_summary, mo):
    mo.md(
        r"""
        ## 3. Scaled training and radius

        The paper reports that scaled cross-entropy improves uniform-path QM9
        validity from 475.4 to 556.0. Our local-count softmax reconstruction moved
        in the opposite direction. Radius one was least harmful, but every scaled
        model generated fewer valid and novel molecules than ordinary
        cross-entropy at 256 Euler evaluations.
        """
    )
    mo.ui.table(loss_summary)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ![Scaled-loss radius ablation](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/radius_ablation.png)

    All four objectives optimized smoothly, so the divergence is not explained
    by an obvious training collapse. Their weighted loss values are not
    numerically comparable.

    ![Training curves](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/images/training_dynamics.png)

    ## 4. Exact scope and interpretation

    - **Data:** public DeepChem QM9, RDKit canonical isomeric SMILES, explicit
      hydrogens removed, strings over 30 tokens excluded.
    - **Model:** 14,375,808-parameter bidirectional masked Transformer; 25,000
      updates, batch 512, quadratic reveal schedule.
    - **Evidence:** four seeds for ordinary cross-entropy and scaled radii 1–3;
      1,024 generations per sampler and NFE; fixed NFE 8–256; temperature
      0.8–1.2 and neighbor-scale robustness checks.
    - **Compute:** Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs, with 16
      concurrently allocated GPUs at peak and 0.5125 hours end-to-end wall time.

    The paper used a 92M-parameter architecture and an unavailable original
    implementation. The reconstructed scaled objective, tokenizer, data split,
    and smaller model are consequential substitutions. The result therefore
    says that this compact setup did not show the reported intervention gains;
    it does not establish that the full-scale claim is incorrect. OpenWebText
    and context-weighted path training were not attempted.

    Formal evidence is stored in
    [`results/primary_generation.csv`](https://raw.githubusercontent.com/alphaXiv/context-weighted-discrete-flow-matching/main/reports/qm9/results/primary_generation.csv),
    with diagnostics, training curves, and robustness records in the same
    public directory. Expensive training is intentionally not rerun here.
    """)
    return


if __name__ == "__main__":
    app.run()
