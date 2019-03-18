import csv
import datapoint


class Dataset:
    def __init__(self, filename, p=False):
        self.positive = list()
        self.negative = list()
        self.read_data(filename, p)

    def read_data(self, filename, p):
        with open(filename) as File:
            reader = csv.reader(File, delimiter=',')
            for row in reader:
                if row[3] == '1':
                    point = datapoint.DataPoint(float(row[1]), float(row[2]), int(row[3]))
                    self.positive.append(point)
                else:
                    if p is True:
                        row[3] = -1
                    point = datapoint.DataPoint(float(row[1]), float(row[2]), int(row[3]))
                    self.negative.append(point)

    def perceptron_data(self):
        r_positive = list()
        for point in self.positive:
            r_positive.append([1, point.x, point.y])
        r_negative = list()
        for point in self.negative:
            r_negative.append([1, point.x, point.y])
        return r_positive, r_negative
