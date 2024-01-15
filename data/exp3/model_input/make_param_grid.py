import csv
import numpy as np

chainNum = 10000
with open('grid_diagnostic.csv', 'w') as csv_file :
    writer = csv.writer(csv_file, delimiter=',')
    for distractorWeight in np.arange(0, 1.1, 0.01):
        writer.writerow(['cdf', 'additive', 1, 0, distractorWeight, chainNum])
        writer.writerow(['cdf', 'RSA',  1,  0, distractorWeight, chainNum])
        chainNum = chainNum + 1
