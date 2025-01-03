# Lexical search and pragmatics in connector

This repository contains the code to reproduce the analyses presented in the following paper:

> Kumar, A.A., Hawkins, R.D. (in press) Lexical search and social reasoning jointly explain communication in associative reference games. *Journal of Experimental Psychology: General*.


## Getting started

1. Create a virtual environment with the required dependencies:

```
conda env create -f environment.yml
conda activate connector
conda env list
```

2. Download the data files from the [OSF repositoy](https://osf.io/xq2s3/) and store them inside the relevant experiment subfolders (e.g., `data/exp1/model_output/`).

3. To reproduce the analyses, run the individual .Rmd files corresponding to each experiment in the `analysis/` folder.
