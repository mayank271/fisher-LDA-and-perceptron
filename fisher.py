import numpy as np
from numpy.linalg import inv
import dataset
import matplotlib.pyplot as plt


class Fisher:
    def __init__(self, filename, train=700):
        self.data = dataset.Dataset(filename, train)
        self.mean_positive, self.mean_negative = self.get_means()
        self.w = self.get_w(self.get_covariance_matrix())
        self.transformed_positive_points = list()
        self.transformed_negative_points = list()
        self.transform_points()
        self.threshold = self.solve()
        self.misclassified = self.test()
        self.plot(filename.split('.')[0]+'_fisher.png')

    def get_means(self):
        """
        :return: means (numpy array) for all positive and negative data points
        """
        mean_x = 0
        mean_y = 0
        for point in self.data.positive:
            mean_x += point.x
            mean_y += point.y
        mean_x /= len(self.data.positive)
        mean_y /= len(self.data.positive)
        mean_positive = np.zeros((2, 1))
        mean_positive[0] = mean_x
        mean_positive[1] = mean_y

        mean_x = 0
        mean_y = 0
        for point in self.data.negative:
            mean_x += point.x
            mean_y += point.y
        mean_x /= len(self.data.negative)
        mean_y /= len(self.data.negative)
        mean_negative = np.zeros((2, 1))
        mean_negative[0] = mean_x
        mean_negative[1] = mean_y
        return mean_positive, mean_negative

    def get_covariance_matrix(self):
        """
        :return: Within class covariance matrix
        """
        variance = np.zeros((2, 2))
        for point in self.data.positive:
            positive_var = np.zeros((2, 1))
            positive_var[0] = point.x - self.mean_positive[0]
            positive_var[1] = point.y - self.mean_positive[1]
            variance += np.dot(positive_var, positive_var.transpose())

        for point in self.data.negative:
            negative_var = np.zeros((2, 1))
            negative_var[0] = point.x - self.mean_negative[0]
            negative_var[1] = point.y - self.mean_negative[1]
            variance += np.dot(negative_var, negative_var.transpose())

        return variance

    def get_w(self, covar):
        """
        W = inverse(Covariance)* Difference of means
        :param covar: Within Class Covariance Matrix
        :return: Weight Matrix
        """
        inverse = np.zeros((2, 2))
        try:
            inverse = inv(covar)
        except:
            print("Error while calculating inverse")

        return np.dot(inverse, (self.mean_negative - self.mean_positive))

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

    def test(self):
        misclassified = list()
        for point in self.data.positive_test:
            p = np.zeros((2, 1))
            p[0] = self.data.positive[0].x
            p[1] = self.data.positive[0].y
            np_point = np.zeros((2, 1))
            np_point[0] = point.x
            np_point[1] = point.y
            val = float((np.dot(self.w.transpose(), np_point)))
            e_val = float((np.dot(self.w.transpose(), p)))
            if e_val > self.threshold:
                if val < self.threshold:
                    misclassified.append(point)
            if e_val < self.threshold:
                if val > self.threshold:
                    misclassified.append(point)
        for point in self.data.negative_test:
            p = np.zeros((2, 1))
            p[0] = self.data.negative[0].x
            p[1] = self.data.negative[0].y
            np_point = np.zeros((2, 1))
            np_point[0] = point.x
            np_point[1] = point.y
            val = float((np.dot(self.w.transpose(), np_point)))
            e_val = float((np.dot(self.w.transpose(), p)))
            if e_val > self.threshold:
                if val < self.threshold:
                    misclassified.append(point)
            if e_val < self.threshold:
                if val > self.threshold:
                    misclassified.append(point)

        return misclassified
