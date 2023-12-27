import csv
import numpy as np

chainNum = 100000
with open('grid.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for alpha in [4, 8, 12, 16] :
        for costWeight in [0.16]:
            writer.writerow(['cdf', 'no_prag', alpha, round(costWeight,2), 0, chainNum])
            for distractorWeight in [0.4, 0.45, 0.5]:
                writer.writerow(['cdf', 'additive', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'additive', alpha,  round(costWeight,2), distractorWeight, chainNum])
                chainNum = chainNum + 1
