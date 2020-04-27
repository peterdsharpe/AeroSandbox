import copy

if __name__ == '__main__':
    epss = np.logspace(-10, -1, 30)
    baseline_objective = augmented_objective(x0)
    xis = []
    for eps in epss:
        xi = copy.copy(x0)
        xi[4] += eps
        xis.append(xi)

    objs = [augmented_objective(xi) for xi in xis]
    # pool = mp.Pool(mp.cpu_count())
    # objs = pool.map(augmented_objective, xis)
    # pool.close()

    objs = np.array(objs)

    derivs = (objs - baseline_objective) / epss

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.loglog(epss, np.abs(derivs), ".-")
    plt.show()