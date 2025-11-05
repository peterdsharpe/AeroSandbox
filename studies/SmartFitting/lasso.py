from data import x, y_data
import aerosandbox as asb
import aerosandbox.numpy as np

degree = 10

opti = asb.Opti()

coeffs = opti.variable(init_guess=np.zeros(degree + 1))

vandermonde = np.ones((len(x), degree + 1))
for j in range(1, degree + 1):
    vandermonde[:, j] = vandermonde[:, j - 1] * x

y_model = vandermonde @ coeffs

error = np.sum((y_model - y_data) ** 2)

abs_coeffs = opti.variable(init_guess=np.zeros(degree + 1))
opti.subject_to([abs_coeffs > coeffs, abs_coeffs > -coeffs])

opti.minimize(error + 1e-4 * np.sum(abs_coeffs))

sol = opti.solve(verbose=False)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)

    x_plot = np.linspace(x[0], x[-1], 100)
    vandermonde_plot = np.ones((len(x_plot), degree + 1))
    for j in range(1, degree + 1):
        vandermonde_plot[:, j] = vandermonde_plot[:, j - 1] * x_plot
    y_plot = vandermonde_plot @ sol(coeffs)

    plt.plot(x, y_data, ".")
    plt.plot(x_plot, sol(y_plot), "-")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.bar(x=np.arange(degree + 1), height=sol(coeffs))
    plt.show()
