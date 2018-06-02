import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def load_data():
    m = 92000
    data = np.load('dataset/dataset.npz')

    x_train = data['arr_0']
    y_train = data['arr_1'].reshape(78200, 1)
    x_test = data['arr_2']
    y_test = data['arr_3'].reshape(m - 78200, 1)

    X = np.vstack([x_train, x_test]).reshape(m, 1024)
    Y = np.vstack([y_train, y_test]).reshape(m, 1)

    return X, Y

def shuffle(X, Y):
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y)
    return X, Y

def plot(X, Y, n):
    plt.imshow(X[n, :].reshape(32, 32), cmap = 'Greys')
    plt.show()
    print(Y[n])

def scale(X, factor = 255):
    return X*(1/255)

def split(x, y, ratio):
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size = ratio)

def main():
    X, Y = load_data()
    X, Y = shuffle(X, Y)
    X = scale(X)

if __name__ == '__main__':
    main()
