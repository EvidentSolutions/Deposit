# Feature Transport Across Grammatical Roles in Pythia-410M

**Olli Tuomi, Evident Solutions Oy**

Cross-position feature transport in transformers has been studied mainly through subject-verb agreement. We measure the same mechanism -- a low-rank linear map from a source token's representation to a consuming position -- across three grammatical roles in Pythia-410M-deduped. Object channels (centered R² 0.52-0.56) are stronger than the subject channel (0.41), despite English having no object agreement. The channel carries semantic content beyond grammatical features (73% unnamed, 7 axes causally validated). The transported copy is decodable from layer 8 but only becomes causal at layers 20-23.

## Repository contents

```
paper.tex              LaTeX source
figures/               PDF and PNG figures (5 figures)
make_figures.py        Regenerate all figures from data/
code/                  14 experiment scripts (Pythia-410M)
data/                  15 JSON result files
CITATION.cff           Citation metadata
```

## Building the paper

```bash
pdflatex paper.tex
pdflatex paper.tex   # twice for references
```

Requires: `mathpazo`, `berasans`, `beramono`, `microtype`, `natbib`, `booktabs`, `caption`, `parskip`, `setspace`, `xcolor`, `hyperref`.

## Regenerating figures

```bash
pip install matplotlib numpy
python make_figures.py
```

Reads from `data/` and writes to `figures/`.

## Running experiments

All scripts require `torch` and `transformers` with access to `EleutherAI/pythia-410m-deduped`.

```bash
pip install torch transformers
python code/role_deposit_test.py          # Table 1: role comparison
python code/centered_r2_all_roles.py      # Table 1: centered R² values
python code/feature_deposit_map.py        # Table 2: feature transport
python code/feature_channel_discover.py   # Channel content (27% named)
python code/sweep_channel.py              # Table 3: discover-validate
python code/place_causal.py               # Place axis causal test
python code/body_causal.py                # Body-part axis causal test
python code/transported_causal.py         # Table 4 + Figure 4: layer sweep
python code/global_structure_control_v2.py  # Global-structure control
python code/position_contribution.py      # Position contribution analysis
python code/position_contribution_long.py # Context-length sweep
python code/layer_pair_sweep.py           # Appendix B: layer-pair heatmap
```

Results are written to `data/`.

## License

MIT
