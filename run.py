import perceptron
import fisher
import time

if __name__ == '__main__':

    fisher_train = 1000
    print("-----------------------------------------------")
    print("Running Fisher Linear Discriminant...")
    print("Dataset 1...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_1.csv", fisher_train)
    print("Number of training points: {}".format(fisher_train))
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 1 is: {}".format(float(f.threshold)))
    print("Number of misclassified points: {}".format(len(f.misclassified)))
    print()

    print("Dataset 2...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_2.csv", fisher_train)
    print("Number of training points: {}".format(fisher_train))
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 1 is: {}".format(float(f.threshold)))
    print("Number of misclassified points: {}".format(len(f.misclassified)))
    print()

    print("Dataset 3...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_3.csv", fisher_train)
    print("Number of training points: {}".format(fisher_train))
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 1 is: {}".format(float(f.threshold)))
    print("Number of misclassified points: {}".format(len(f.misclassified)))
    print()

    learning_param = 0.1
    iterations = 1000
    percetron_train = 700
    print("-----------------------------------------------")
    print("Running Perceptron...")
    print("Dataset 1...")

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_1.csv", learning_param, iterations, percetron_train)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print("Number of misclassified points: {}".format(len(p.misclassified)))
    print()

    print("Dataset 2...")

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_2.csv", learning_param, iterations)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print("Number of misclassified points: {}".format(len(p.misclassified)))
    print()

    print("Dataset 3...")

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_3.csv", learning_param, iterations)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print("Number of misclassified points: {}".format(len(p.misclassified)))
    print()
