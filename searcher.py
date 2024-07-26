import numpy as np
import csv


class Searcher:
    def __init__(self, indexPath):
        # store our index path
        self.indexPath = indexPath
        
    def search(self, queryFeatures):
        # initialize our dictionary of results
        results = {}
        
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)

                results[row[0]] = d

            f.close()
        
        return results

    
    def chi2_distance(self, histA, histB, eps = 1e-10):
        # chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        
        max_possible_score = 1  # This is a typical normalization factor
        similarity_percentage = (1 - d / max_possible_score) * 100
        print(similarity_percentage)
        
        return d