import hashlib
import aerosandbox.numpy as np


def eng_string(
        x: float,
        unit: str = None,
        format='%.3g',
        si=True
) -> str:
    '''
    Taken from: https://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6/40691220

    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    Args:

        x: The value to be formatted. Float or int.

        unit: A unit of the quantity to be expressed, given as a string. Example: Newtons -> "N"

        format: A printf-style string used to format the value before the exponent.

        si: if true, use SI suffix for exponent. (k instead of e3, n instead of
            e-9, etc.)

    Examples:

    With format='%.2f':
        1.23e-08 -> 12.30e-9
             123 -> 123.00
          1230.0 -> 1.23e3
      -1230000.0 -> -1.23e6

    With si=True:
          1230.0 -> "1.23k"
      -1230000.0 -> "-1.23M"

    With unit="N" and si=True:
          1230.0 -> "1.23 kN"
      -1230000.0 -> "-1.23 MN"
    '''
    sign = ''
    if x < 0:
        x = -x
        sign = '-'
    elif x == 0:
        return format % 0
    elif np.isnan(x):
        return "NaN"
    exp = int(np.floor(np.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnμm kMGTPEZY'[(exp3 + 24) // 3]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = f'e{exp3}'

    if unit is not None:
        if si:
            exp3_text = " " + exp3_text + unit
        else:
            exp3_text = exp3_text + " " + unit

    return ('%s' + format + '%s') % (sign, x3, exp3_text)


def latex_sci_notation_string(
        x: float,
        format='%.2e',
) -> str:
    """
    Converts a floating-point number to a LaTeX-style formatted string. Does not include the `$$` wrapping to put you in math mode.

    Does not use scientific notation if the base would be zero.

    Examples:

        latex_sci_notation_string(3000) -> '3 \\times 10^{3}'
    """
    float_str = format % x
    base, exponent = float_str.split("e")
    exponent = int(exponent)
    if exponent == 0:
        return base
    else:
        return r"{0} \times 10^{{{1}}}".format(base, exponent)


def hash_string(string: str) -> int:
    """
    Hashes a string into a quasi-random 32-bit integer! (Based on an MD5 checksum algorithm.)

    Usual warnings apply: it's MD5, don't use this for anything intended to be cryptographically secure.
    """
    md5 = hashlib.md5(string.encode('utf-8'))
    hash_hex = md5.hexdigest()
    hash_int = int(hash_hex, 16)
    hash_int64 = hash_int % (2 ** 32)
    return hash_int64


def trim_string(string: str, length: int = 80) -> str:
    """
    Trims a string to be less than a given length. If the string would exceed the length, makes it end in "...".
    """
    if len(string) > length:
        return string[:length - 3] + "..."
    else:
        return string


def has_balanced_parentheses(string: str, left="(", right=")"):
    parenthesis_level = 0

    for char in string:
        if char == "(":
            parenthesis_level += 1
        elif char == ")":
            parenthesis_level -= 1

    return parenthesis_level == 0
