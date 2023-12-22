# parallel --bar --colsep ',' "sh ./run_parallel.sh {1} {2} {3} {4} {5} {6}" :::: ../data/exp2/model_input//grid.csv
python pragmatics.py $1 $2 $3 $4 $5 $6
