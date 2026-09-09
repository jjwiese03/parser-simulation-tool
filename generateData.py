# generate random DiscConfiguration
import numpy as np 
import json


def generateDiscConfiguration():
    n = np.random.randint(1, 20)
    dist = np.random.uniform(0, 5, n).tolist()
    widths = np.random.uniform(0, 0.8, n).tolist()

    pos = []
    for i in range(n):
        if i == 0:
            pos.append(dist[i])
        else:
            pos.append(dist[i] + widths[i-1] + pos[i-1])

    return {
        "n": n,
        "distances": dist,
        "positions": pos,
        "widths": widths
    }


class DataGenerator:
    '''
    This class generates random DiscConfiguration data in different formats together with its labels.

    labels are generated as a list, where each label is for a Number in the formated Data. 

    LABEL:
    0 = Undefined (not a information about a disc)
    1 = Distance 
    2 = Position
    3 = Width

    '''
    def __init__(self):
        self.formats = ["JSON", "CSV"]

    def generate(self, filename, sampleSize=32, formats=None):
        if formats is None:
            formats = self.formats
        with open(filename, "w") as f:
            f.write("[")
            for i in range(sampleSize):
                config = generateDiscConfiguration()

                match np.random.choice(formats):
                    case "JSON":
                        encoding = self.JSONEncoder().encode(config)
                        label = self.JSONEncoder().labels(config)

                    case "CSV":
                        encoding = self.CSVEncoder().encode(config)
                        label = self.CSVEncoder().labels(config)

                f.write(str({'data': encoding, 'label': label}))
                if i < sampleSize - 1:
                    f.write(",")
            f.write("]")
        return 
    
    class JSONEncoder:
        def encode(self, DiscData):
            return json.dumps(DiscData, indent=4)
        def labels(self, DiscData):
            return [1]*DiscData["n"] + [2]*DiscData["n"] + [3]*DiscData["n"]

    class CSVEncoder:
        def encode(self, DiscData):
            n = DiscData["n"]
            distances = DiscData["distances"]
            positions = DiscData["positions"]
            widths = DiscData["widths"]

            csv_data = "distances,positions,widths\n"
            for i in range(n):
                csv_data += f"{distances[i]},{positions[i]},{widths[i]}\n"
            return csv_data

        def labels(self, DiscData):
            return [1,2,3] * DiscData["n"]
      
Generator = DataGenerator()

Generator.generate("train_data.txt", sampleSize=200, formats=["JSON", "CSV"])