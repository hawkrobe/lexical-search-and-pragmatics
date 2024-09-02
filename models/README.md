# Experiment 1

Run the model with a specific parameter setting as follows:

``` sh
python pragmatics.py --experiment exp1 --cost_type 'cdf' --inf_type 'RSA' --alpha 12 --costweight 0.32
```

To generate `speaker_df.csv` containing clue production probabilities, run the grid search using

``` sh
parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} exp1" :::: ../data/exp1/model_input/grid.csv
```

To generate a list of examples (e.g. Table 1), run

``` sh
python pragmatics.py --examples --experiment exp1 --cost_type 'cdf' --inf_type 'RSA' --alpha 12 --costweight 0.32
```

# Experiment 2

To generate model comparison metrics, run:

`python pragmatics.py`

# Appendix A

The lexical search model with top-down diagnosticity influences is implemented in `blended.py`.

# Appendix B

The lexical search models taking intersections and unions are implemented in `baselines.py`.
