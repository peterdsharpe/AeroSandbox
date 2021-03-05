import math
import aerosandbox.numpy as np
import hashlib


def eng_string(x: float, format='%.3g', si=True):
    '''
    Taken from: https://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6/40691220

    Returns float/int value <x> formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.

    format: printf-style string used to format the value before the exponent.

    si: if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
    e-9 etc.

    E.g. with format='%.2f':
        1.23e-08 -> 12.30e-9
             123 -> 123.00
          1230.0 -> 1.23e3
      -1230000.0 -> -1.23e6

    and with si=True:
          1230.0 -> 1.23k
      -1230000.0 -> -1.23M
    '''
    sign = ''
    if x < 0:
        x = -x
        sign = '-'
    exp = int(math.floor(math.log10(x)))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = 'yzafpnum kMGTPEZY'[(exp3 + 24) // 3]
    elif exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    return ('%s' + format + '%s') % (sign, x3, exp3_text)


def hash_string(string: str) -> int:
    """
    Hashes a string into a quasi-random 32-bit integer! (Based on an MD5 checksum algorithm.)
    """
    md5 = hashlib.md5(string.encode('utf-8'))
    hash_hex = md5.hexdigest()
    hash_int = int(hash_hex, 16)
    hash_int64 = hash_int % (2 ** 32)
    return hash_int64
