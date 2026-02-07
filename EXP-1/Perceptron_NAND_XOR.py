import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

datasets = {
    "NAND": np.array([1, 1, 1, 0]),
    "XOR":  np.array([0, 1, 1, 0])
}

lr = 0.1
max_epochs = 20

def train(name, y):
    w = np.zeros(3)

    def predict(x):
        return 1 if np.dot(w, x) >= 0 else 0

    plt.ion()
    print(f"\nTraining {name} gate")

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}")
        errors = 0

        for xi, target in zip(X, y):
            y_hat = predict(xi)
            error = target - y_hat

            if error != 0:
                w += lr * error * xi
                errors += 1

                print(f"w = [{w[0]: .3f}, {w[1]: .3f}, {w[2]: .3f}]")

                preds = np.array([predict(x) for x in X])
                correct = preds == y

                plt.clf()
                plt.title(f"{name} | epoch {epoch+1}")

                plt.scatter(X[correct, 1], X[correct, 2], c="green", s=100)
                plt.scatter(X[~correct, 1], X[~correct, 2], c="red", s=100)

                x_vals = np.array([-1, 2])
                if w[2] != 0:
                    y_vals = -(w[0] + w[1]*x_vals) / w[2]
                    plt.plot(x_vals, y_vals, "k-")

                plt.axhline(0)
                plt.axvline(0)
                plt.xlim(-1, 2)
                plt.ylim(-1, 2)
                plt.pause(0.4)

        print(f"errors this epoch = {errors}")
        print("-" * 30)

        if errors == 0:
            print("Converged")
            break
    else:
       print("Did not converge (dataset not linearly separable)")


    plt.ioff()
    plt.show()
    print(f"final w = [{w[0]: .3f}, {w[1]: .3f}, {w[2]: .3f}]")


for name, y in datasets.items():
    train(name, y)
