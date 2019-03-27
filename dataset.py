import csv
import datapoint
from random import shuffle


class Dataset:
    def __init__(self, filename, train, p=False):
        self.positive = list()
        self.negative = list()
        self.positive_test = list()
        self.negative_test = list()
        self.read_data(filename, p, train)

    def read_data(self, filename, p, train):
        positive_temp = list()
        negative_temp = list()
        all_temp = list()
        with open(filename) as File:
            reader = csv.reader(File, delimiter=',')
            for row in reader:
                if row[3] == '1':
                    point = datapoint.DataPoint(float(row[1]), float(row[2]), int(row[3]))
                    positive_temp.append(point)
                    all_temp.append(point)
                else:
                    if p is True:
                        row[3] = -1
                    point = datapoint.DataPoint(float(row[1]), float(row[2]), int(row[3]))
                    negative_temp.append(point)
                    all_temp.append(point)
        shuffle(all_temp)
        train_temp = all_temp[:train]
        test_temp = all_temp[train:]
        self.positive = list(set(positive_temp) & set(train_temp))
        self.positive_test = list(set(positive_temp) & set(test_temp))
        self.negative = list(set(negative_temp) & set(train_temp))
        self.negative_test = list(set(negative_temp) & set(test_temp))

    def perceptron_data(self):
        r_positive = list()
        for point in self.positive:
            r_positive.append([1, point.x, point.y])
        r_negative = list()
        for point in self.negative:
            r_negative.append([1, point.x, point.y])
        return r_positive, r_negative
