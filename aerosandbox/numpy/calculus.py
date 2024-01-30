import numpy as _onp
import casadi as _cas
from aerosandbox.numpy.determine_type import is_casadi_type
from aerosandbox.numpy.arithmetic_dyadic import centered_mod as _centered_mod
from aerosandbox.numpy.array import array, concatenate, reshape


def diff(a, n=1, axis=-1, period=None):
    """
    Calculate the n-th discrete difference along the given axis.

    See syntax here: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
    """
    if period is not None:
        return _centered_mod(
            diff(a, n=n, axis=axis),
            period
        )

    if not is_casadi_type(a):
        return _onp.diff(a, n=n, axis=axis)

    else:
        if axis != -1:
            raise NotImplementedError("This could be implemented, but haven't had the need yet.")

        result = a
        for i in range(n):
            result = _cas.diff(a)
        return result


def gradient(
        f,
        *varargs,
        axis=None,
        edge_order=1,
        n=1,
):
    if (
            not is_casadi_type(f)
            and all([not is_casadi_type(vararg) for vararg in varargs])
            and n == 1
    ):
        return _onp.gradient(
            f,
            *varargs,
            axis=axis,
            edge_order=edge_order
        )
    else:
        f = array(f)
        shape = f.shape

        # Handle the varargs argument
        if len(varargs) == 0:
            varargs = (1.,)

        if len(varargs) == 1:
            varargs = [varargs[0] for i in range(len(shape))]

        if len(varargs) != len(shape):
            raise ValueError("You must specify either 0, 1, or N varargs, where N is the number of dimensions of f.")
        else:
            dxes = []

            for i, vararg in enumerate(varargs):
                if _onp.prod(array(vararg).shape) == 1:  # If it's a scalar, you have dx values
                    dxes.append(
                        vararg * _onp.ones(shape[i] - 1)
                    )
                else:
                    dxes.append(
                        diff(vararg)
                    )

        # Handle the axis argument, with the edge case the CasADi arrays are always 2D
        if axis is None:
            if is_casadi_type(f, recursive=False) and shape[1] == 1:
                axis = 0
            elif len(shape) <= 1:
                axis = 0
            else:
                axis = tuple(_onp.arange(len(shape)))

        try:
            tuple(axis)  # See if axis is iterable
            axis_is_iterable = True
        except TypeError:
            axis_is_iterable = False

        if axis_is_iterable:
            return [
                gradient(
                    f,
                    varargs[axis_i],
                    axis=axis_i,
                    edge_order=edge_order
                )
                for axis_i in axis
            ]

        else:
            # Check validity of axis
            if axis < 0:
                axis = len(shape) + axis
            if is_casadi_type(f, recursive=False):
                if axis not in [0, 1]:
                    raise ValueError("axis must be 0 or 1 for CasADi arrays.")

            dx = dxes[axis]
            dx_shape = [1] * len(shape)
            dx_shape[axis] = shape[axis] - 1
            dx = reshape(dx, dx_shape)

            def get_slice(slice_obj: slice):
                slices = [slice(None)] * len(shape)
                slices[axis] = slice_obj
                return tuple(slices)

            hm = dx[get_slice(slice(None, -1))]
            hp = dx[get_slice(slice(1, None))]

            dfp = (
                    f[get_slice(slice(2, None))]
                    - f[get_slice(slice(1, -1))]
            )
            dfm = (
                    f[get_slice(slice(1, -1))]
                    - f[get_slice(slice(None, -2))]
            )

            if n == 1:
                grad_f = (
                                 hm ** 2 * dfp + hp ** 2 * dfm
                         ) / (
                                 hm * hp * (hm + hp)
                         )

                if edge_order == 1:
                    # First point
                    df_f = dfm[get_slice(slice(0, 1))]
                    h_f = hm[get_slice(slice(0, 1))]
                    grad_f_first = df_f / h_f

                    # Last point
                    df_l = dfp[get_slice(slice(-1, None))]
                    h_l = hp[get_slice(slice(-1, None))]
                    grad_f_last = df_l / h_l

                elif edge_order == 2:
                    # First point
                    dfm_f = dfm[get_slice(slice(0, 1))]
                    dfp_f = dfp[get_slice(slice(0, 1))]
                    hm_f = hm[get_slice(slice(0, 1))]
                    hp_f = hp[get_slice(slice(0, 1))]
                    grad_f_first = (
                                           2 * dfm_f * hm_f * hp_f + dfm_f * hp_f ** 2 - dfp_f * hm_f ** 2
                                   ) / (
                                           hm_f * hp_f * (hm_f + hp_f)
                                   )

                    # Last point
                    dfm_l = dfm[get_slice(slice(-1, None))]
                    dfp_l = dfp[get_slice(slice(-1, None))]
                    hm_l = hm[get_slice(slice(-1, None))]
                    hp_l = hp[get_slice(slice(-1, None))]
                    grad_f_last = (
                                          -dfm_l * hp_l ** 2 + dfp_l * hm_l ** 2 + 2 * dfp_l * hm_l * hp_l
                                  ) / (
                                          hm_l * hp_l * (hm_l + hp_l)
                                  )

                else:
                    raise ValueError("Invalid edge_order.")

                grad_f = concatenate((
                    grad_f_first,
                    grad_f,
                    grad_f_last
                ), axis=axis)

                return grad_f

            elif n == 2:
                grad_grad_f = (
                        2 / (hm + hp) * (
                        dfp / hp - dfm / hm
                )
                )

                grad_grad_f_first = grad_grad_f[get_slice(slice(0, 1))]
                grad_grad_f_last = grad_grad_f[get_slice(slice(-1, None))]

                grad_grad_f = concatenate((
                    grad_grad_f_first,
                    grad_grad_f,
                    grad_grad_f_last
                ), axis=axis)

                return grad_grad_f

            else:
                raise ValueError(
                    "A second-order reconstructor only supports first derivatives (n=1) and second derivatives (n=2).")


def trapz(x, modify_endpoints=False):  # TODO unify with NumPy trapz, this is different
    """
    Computes each piece of the approximate integral of `x` via the trapezoidal method with unit spacing.
    Can be viewed as the opposite of diff().

    Args:
        x: The vector-like object (1D np.ndarray, cas.MX) to be integrated.

    Returns: A vector of length N-1 with each piece corresponding to the mean value of the function on the interval
        starting at index i.

    """
    import warnings
    warnings.warn(
        "trapz() will eventually be deprecated, since NumPy plans to remove it in the upcoming NumPy 2.0 release (2024). \n"
        "For discrete intervals, use asb.numpy.integrate_discrete_intervals(f, method=\"trapz\") instead.",
        PendingDeprecationWarning)

    integral = (
                       x[1:] + x[:-1]
               ) / 2
    if modify_endpoints:
        integral[0] = integral[0] + x[0] * 0.5
        integral[-1] = integral[-1] + x[-1] * 0.5

    return integral


if __name__ == '__main__':
    import aerosandbox as asb
    import aerosandbox.numpy as np

    import casadi as cas

    print(diff(cas.DM([355, 5]), period=360))
    #
    # # a = np.linspace(-500, 500, 21) % 360 - 180
    # # print(diff(a, period=360))
    #
    # x = np.cumsum(np.arange(10))
    # y = x ** 2
    #
    # print(gradient(y, x, edge_order=1))
    # print(gradient(y, x, edge_order=1))
    # print(gradient(y, x, edge_order=1, n=2))
    #
    # opti = asb.Opti()
    # x = opti.variable(init_guess=[355, 5])
    # d = diff(x, period=360)
    # opti.subject_to([
    #     # x[0] == 3,
    #     x[0] > 180,
    #     x[1] < 180,
    #     d < 20,
    #     d > -20
    # ])
    # opti.maximize(np.sum(np.cosd(x)))
    # sol = opti.solve(
    #     behavior_on_failure="return_last"
    # )
    # print(sol.value(x))