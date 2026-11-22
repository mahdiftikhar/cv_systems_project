# Final report: data, figures and build

The LaTeX source of the final report is in `report/`, on the CVPR 2027 template
(release `CVPR2027-v1` of https://github.com/cvpr-org/author-kit). These scripts produce every
number and figure it prints.

1. `make_data.py` reads the saved run folders, the analysis tables under `experiment/sweeps/` and
   the model checkpoints, and writes `report/generated/numbers.tex` (one macro per number),
   `report/generated/jobs_*.tex` (the appendix tables) and `figure_data.json`. The run folders are
   git-ignored, so this step runs only on the machine that holds them. Set `PROJECT_ROOT` to point
   it at another checkout.
2. `make_figures.py` draws `report/figures/*.pdf` from `figure_data.json`. Its style follows the
   helpers of https://github.com/ppriyank/plot.
3. `cd report && latexmk -pdf main.tex` builds `main.pdf`.

```bash
python -I report_scripts/make_data.py
python -I report_scripts/make_figures.py
cd report && latexmk -pdf main.tex
```

The LLaVA-1.6 values come from `llava_teammates.json`, transcribed from the slides of the
teammates who ran those experiments, because their run files are not in this repository.
The report shows no manga page, narration or generated summary, only scores, counts and token IDs.
