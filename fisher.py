import numpy as np
from numpy.linalg import inv
import dataset
import matplotlib.pyplot as plt


class Fisher:
    def __init__(self, filename):
        self.data = dataset.Dataset(filename)
        self.median_positive, self.median_negative = self.get_medians()
        self.w = self.get_w(self.get_covariance_matrix())
        self.transformed_positive_points = list()
        self.transformed_negative_points = list()
        self.transform_points()
        self.threshold = self.solve()
        self.plot(filename.split('.')[0]+'_fisher.png')
        pass

    def get_medians(self):
        """
        :return: Medians (numpy array) for all positive and negative data points
        """
        median_x = 0
        median_y = 0
        for point in self.data.positive:
            median_x += point.x
            median_y += point.y
        median_x /= len(self.data.positive)
        median_y /= len(self.data.positive)
        median_positive = np.zeros((2, 1))
        median_positive[0] = median_x
        median_positive[1] = median_y

        median_x = 0
        median_y = 0
        for point in self.data.negative:
            median_x += point.x
            median_y += point.y
        median_x /= len(self.data.negative)
        median_y /= len(self.data.negative)
        median_negative = np.zeros((2, 1))
        median_negative[0] = median_x
        median_negative[1] = median_y

        return median_positive, median_negative

    def get_covariance_matrix(self):
        """
        :return: Within class covariance matrix
        """
        variance = np.zeros((2, 2))
        for point in self.data.positive:
            positive_var = np.zeros((2, 1))
            positive_var[0] = point.x - self.median_positive[0]
            positive_var[1] = point.y - self.median_positive[1]
            variance += np.dot(positive_var, positive_var.transpose())

        for point in self.data.negative:
            negative_var = np.zeros((2, 1))
            negative_var[0] = point.x - self.median_negative[0]
            negative_var[1] = point.y - self.median_negative[1]
            variance += np.dot(negative_var, negative_var.transpose())

        return variance

    def get_w(self, covar):
        """
        W = inverse(Covariance)* Difference of Medians
        :param covar: Within Class Covariance Matrix
        :return: Weight Matrix
        """
        inverse = np.zeros((2, 2))
        try:
            inverse = inv(covar)
        except:
            print("Error while calculating inverse")

        return np.dot(inverse, (self.median_negative - self.median_positive))

    def transform_points(self):
        """
        Reduces all points to 1-Dimension
        """
        for point in self.data.positive:
            np_point = np.zeros((2, 1))
            np_point[0] = point.x
            np_point[1] = point.y
            self.transformed_positive_points.append(float((np.dot(self.w.transpose(), np_point))))
        for point in self.data.negative:
            np_point = np.zeros((2, 1))
            np_point[0] = point.x
            np_point[1] = point.y
            self.transformed_negative_points.append(float((np.dot(self.w.transpose(), np_point))))

    def plot(self, filename):
        """
        Plots the points in 1-Dimensions along with the normal distributions for both classes
        :param filename: To save the plot (.png)
        :return:
        """
        points = sorted(self.transformed_negative_points + self.transformed_positive_points)
        plt.scatter([point for point in self.transformed_positive_points],
                    [0 for i in range(len(self.transformed_positive_points))], color='red')
        plt.scatter([point for point in self.transformed_negative_points],
                    [0 for i in range(len(self.transformed_negative_points))], color='blue')
        plt.plot(points, 1 / (np.std(self.transformed_positive_points) * np.sqrt(2 * np.pi)) *
                 np.exp(- (points - np.mean(self.transformed_positive_points)) ** 2 /
                        (2 * np.std(self.transformed_positive_points) ** 2)), linewidth=1, color='r')
        plt.plot(points, 1 / (np.std(self.transformed_negative_points) * np.sqrt(2 * np.pi)) *
                 np.exp(- (points - np.mean(self.transformed_negative_points)) ** 2 /
                        (2 * np.std(self.transformed_negative_points) ** 2)), linewidth=1, color='b')
        plt.savefig(filename)
        plt.show()

    def solve(self):
        """
        To solve the two normal distributions to get threshold
        :return: Threshold Value
        """
        a = 1 / (2 * np.std(self.transformed_positive_points) **
                 2) - 1 / (2 * np.std(self.transformed_positive_points) ** 2)
        b = np.mean(self.transformed_negative_points) / \
            (np.std(self.transformed_positive_points) ** 2) - np.mean(self.transformed_positive_points) /\
            (np.std(self.transformed_positive_points) ** 2)
        c = np.mean(self.transformed_positive_points) ** 2 /\
            (2 * np.std(self.transformed_positive_points) ** 2) - np.mean(self.transformed_negative_points) ** 2 /\
            (2 * np.std(self.transformed_positive_points) ** 2) - np.log(np.std(self.transformed_positive_points) /
                                                                         np.std(self.transformed_positive_points))
        return np.roots([a, b, c])
