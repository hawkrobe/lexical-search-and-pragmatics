import csv
import numpy as np

chainNum = 10000
with open('grid.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for alpha in [2, 4, 8, 12, 16, 32] :
        for costWeight in [0, 0.01, 0.02, 0.04, 0.08, .16, .32]:
            writer.writerow(['cdf', 'no_prag', alpha, round(costWeight,2), 0, chainNum])
            for distractorWeight in [0, 0.001, 0.01, 0.02, 0.04]:
                writer.writerow(['cdf', 'additive', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'additive', alpha,  round(costWeight,2), distractorWeight, chainNum])
                chainNum = chainNum + 1
