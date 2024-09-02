# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5}" :::: ../data/exp1/model_input/grid.csv
python3 pragmatics.py --cost_type $1 --inf_type $2 --alpha $3 --costweight $4 --pid $5
