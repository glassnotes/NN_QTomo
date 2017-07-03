# Extract data about the angles and output to separate CSV file

import sys
import csv
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Run as python angle_extractor.py infile outfile")
    infile, outfile = sys.argv[1], sys.argv[2]

    # Gather all results
    test_angles = []
    pred_results = []
    closest_results = []
    lbmle_results = [] 

    with open(infile, 'r') as incsvfile:
        filereader = csv.reader(incsvfile, delimiter=",")
        for row in filereader:
            if row[0] == "test":
                test_angles.append(np.arccos(float(row[3])) / np.pi)
            elif row[0] == "pred":
                pred_results.append(float(row[-1]))
            elif row[0] == "closest":
                closest_results.append(float(row[-1]))
            elif row[0] == "lbmle":
                lbmle_results.append(float(row[-1]))

    header = ["angle", "pred", "closest", "lbmle"]

    with open(outfile, 'w') as outcsvfile:
        filewriter = csv.writer(outcsvfile)
        filewriter.writerow(header)
        for row in zip(test_angles, pred_results, closest_results, lbmle_results):
            filewriter.writerow(row) 

if __name__ == "__main__":
    main()

    
