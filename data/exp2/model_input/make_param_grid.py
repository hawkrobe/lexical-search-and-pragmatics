import csv
import numpy as np

chainNum = 0
with open('grid.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for alpha in [4, 6, 8, 10, 12, 14, 18, 20, 22, 24] :
        for costWeight in [0, 0.01,0.02,0.04,0.08, .1, .12, .14, .18, .2, .22, .24, .26, .28, .30, 0.32, 0.64, 1]:
            writer.writerow(['cdf', 'prag', alpha, round(costWeight,2), chainNum])
            writer.writerow(['cdf', 'no_prag', alpha, round(costWeight,2), chainNum])
            writer.writerow(['freq', 'prag', alpha,  round(costWeight,2), chainNum])
            chainNum = chainNum + 1
