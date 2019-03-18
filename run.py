import perceptron
import fisher
import time

if __name__ == '__main__':
    print("-----------------------------------------------")
    print("Running Fisher Linear Discriminant...")
    print("Dataset 1...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_1.csv")
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 1 is: {}".format(float(f.threshold)))
    print()

    print("Dataset 2...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_2.csv")
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 2 is: {}".format(float(f.threshold)))
    print()

    print("Dataset 3...")
    start = time.time()
    f = fisher.Fisher("datasets\dataset_3.csv")
    print("Time taken: {}".format(time.time() - start))
    print("W vector: {}".format(f.w))
    print("Threshold value for Dataset 3 is: {}".format(float(f.threshold)))
    print()

    print("-----------------------------------------------")
    print("Running Perceptron...")
    print("Dataset 1...")

    learning_param = 0.1
    iterations = 20

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_1.csv", learning_param, iterations)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print()

    print("Dataset 2...")

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_2.csv", learning_param, iterations)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print()

    print("Dataset 3...")

    start = time.time()
    p = perceptron.Perceptron("datasets\dataset_3.csv", learning_param, iterations)
    print("Time Taken: {}".format(time.time() - start))
    print("W vector: {}".format(p.w))
    print()
