import aerosandbox as asb
import aerosandbox.numpy as np

np.random.seed(0)

x = np.linspace(0, 10, 10)


def f(x):
    return np.sin(x) + 2 * x + 3


y_data = f(x) + 0.1 * np.random.randn(len(x))

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(x, y_data, ".")
    plt.plot(x, f(x), "-")
    plt.show()
