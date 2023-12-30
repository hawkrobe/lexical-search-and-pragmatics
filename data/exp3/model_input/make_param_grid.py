import csv
import numpy as np

chainNum = 10000
with open('grid_diagnostic.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for alpha in [2, 4, 8, 12, 16, 32] :
        for costWeight in np.arange(0, 1.1, 0.1):
            for distractorWeight in np.arange(0, 1.1, 0.1):
                writer.writerow(['cdf', 'no_prag', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'no_prag', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['cdf', 'additive', alpha, round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'additive', alpha,  round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['cdf', 'RSA',  alpha,  round(costWeight,2), distractorWeight, chainNum])
                writer.writerow(['freq', 'RSA',  alpha,  round(costWeight,2), distractorWeight, chainNum])
                chainNum = chainNum + 1
