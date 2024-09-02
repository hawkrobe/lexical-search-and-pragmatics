# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} exp1" :::: ../data/exp1/model_input/grid.csv
# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} exp2" :::: ../data/exp2/model_input/grid.csv
# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} exp3" :::: ../data/exp3/model_input/grid.csv
# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} exp4" :::: ../data/exp4/model_input/grid.csv
python3 pragmatics.py --cost_type $1 --inf_type $2 --alpha $3 --costweight $4 --pid $5 --experiment $6
