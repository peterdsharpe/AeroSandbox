import aerosandbox.numpy as np
import pytest


def test_softmax(plot=False):
    # Test softmax
    x = np.linspace(-10, 10, 100)
    y1 = x
    y2 = -2 * x - 3
    hardness = 0.5

    y_soft = np.softmax(y1, y2, hardness=hardness)

    assert np.softmax(0, 0, hardness=1) == np.log(2)

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set(font_scale=1)

        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
        plt.plot(x, y1, label="y1")
        plt.plot(x, y2, label="y2")
        plt.plot(x, y_soft, label="softmax")
        plt.xlabel(r"x")
        plt.ylabel(r"y")
        plt.title(r"Softmax")
        plt.tight_layout()
        plt.legend()
        plt.show()


def test_sigmoid_types():
    """All documented sigmoid_type options should be accepted (regression
    test: 'logistic' used to raise ValueError due to `== ("tanh" or
    "logistic")`)."""
    for sigmoid_type in ["tanh", "logistic", "arctan", "polynomial"]:
        assert np.sigmoid(0, sigmoid_type=sigmoid_type) == pytest.approx(0.5)
        assert np.sigmoid(1e9, sigmoid_type=sigmoid_type) == pytest.approx(1, abs=1e-6)
        assert np.sigmoid(-1e9, sigmoid_type=sigmoid_type) == pytest.approx(0, abs=1e-6)

    # "tanh" and "logistic" are documented as the same thing.
    x = np.linspace(-3, 3, 7)
    assert np.sigmoid(x, sigmoid_type="logistic") == pytest.approx(
        np.sigmoid(x, sigmoid_type="tanh")
    )

    with pytest.raises(ValueError):
        np.sigmoid(0, sigmoid_type="not_a_real_sigmoid")


if __name__ == "__main__":
    pytest.main()
