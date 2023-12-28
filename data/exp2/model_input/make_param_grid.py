import csv
import numpy as np

chainNum = 100000
with open('grid.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for alpha in [1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32] :
        for costWeight in [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.16, 0.20, 0.32, 0.36, 0.40, 0.64, 1]:
            writer.writerow(['cdf', 'no_prag', alpha, round(costWeight,2), 0, chainNum])
            for distractorWeight in [0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.2, 0.32, 0.36, 0.4, 0.45, 0.64, 1]:
                writer.writerow(['cdf', 'additive', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'additive', alpha,  round(costWeight,2), distractorWeight, chainNum])
                chainNum = chainNum + 1
