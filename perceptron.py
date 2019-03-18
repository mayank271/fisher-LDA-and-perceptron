import numpy as np
import dataset
import random
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, filename, learning_param, iterations):
        self.data = dataset.Dataset(filename, True)
        self.positive, self.negative = self.data.perceptron_data()
        self.w = np.random.rand(3, 1)
        self.learning_parameter = learning_param
        self.iterations = iterations
        self.gradient_descent(self.iterations, filename.split('.')[0]+str(learning_param)+' '+str(iterations))

    def get_misclassified_points(self):
        """
        :return: List of wrongly classified points
        """
        m = list()
        for point in self.positive:
            if np.dot(self.w.transpose(), point) < 0:
                m.append(point)
        for point in self.negative:
            if np.dot(self.w.transpose(), point) > 0:
                m.append(point)
        return m

    def gradient_descent(self, iterations, filename):
        misclassified = None
        curr_point = None
        while iterations > 0:
            try:
                misclassified = self.get_misclassified_points()
                curr_point = random.choice(misclassified)
                if curr_point in self.positive:
                    self.w[0] += self.learning_parameter * curr_point[0]
                    self.w[1] += self.learning_parameter * curr_point[1]
                    self.w[2] += self.learning_parameter * curr_point[2]
                else:
                    self.w[0] -= self.learning_parameter * curr_point[0]
                    self.w[1] -= self.learning_parameter * curr_point[1]
                    self.w[2] -= self.learning_parameter * curr_point[2]
                self.w /= (self.w[0] ** 2 + self.w[1] ** 2 + self.w[2] ** 2) ** 0.5
                self.plot(misclassified, curr_point, filename + ' ' + str(self.iterations - iterations) + '_p.png')
                iterations -= 1
            except:
                iterations = 0
                pass
        self.plot(misclassified, curr_point, filename + ' ' + str(self.iterations - iterations) + '_p.png')

    def plot(self, mis, p, filename):
        mis_x = [m[1] for m in mis]
        mis_y = [m[2] for m in mis]
        positive_x = [point.x for point in self.data.positive]
        positive_y = [point.y for point in self.data.positive]
        negative_x = [point.x for point in self.data.negative]
        negative_y = [point.y for point in self.data.negative]
        plt.scatter(positive_x, positive_y, color='red')
        plt.scatter(negative_x, negative_y, color='blue')
        plt.scatter(mis_x, mis_y, color='green')
        plt.scatter([p[1]], [p[2]], color='red')
        all_x = sorted(positive_x + positive_y)
        all_x = [all_x[0], all_x[len(all_x)-1]]
        plt.plot(all_x, -(self.w[1]/self.w[2]*all_x) - self.w[0]/self.w[2], linewidth=1, color='y')
        plt.savefig(filename)
        plt.show()
