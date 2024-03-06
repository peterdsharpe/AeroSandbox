import hashlib
import aerosandbox.numpy as np


def eng_string(
        x: float,
        unit: str = "",
        format='%.3g',
        si=True,
        add_space_after_number: bool = None,
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

    if si and exp3 >= -24 and exp3 <= 24:
        if exp3 == 0:
            suffix = ""
        else:
            suffix = 'yzafpnμm kMGTPEZY'[(exp3 + 24) // 3]

        if add_space_after_number is None:
            add_space_after_number = (unit != "")

        if add_space_after_number:
            suffix = " " + suffix + unit
        else:
            suffix = suffix + unit

    else:
        suffix = f'e{exp3}'

        if add_space_after_number:
            add_space_after_number = (unit != "")

        if add_space_after_number:
            suffix = suffix + " " + unit
        else:
            suffix = suffix + unit

    return f"{sign}{format % x3}{suffix}"


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
    Trims a string to be less than a given length. If the string would exceed the length, makes it end in ellipses ("…").

    Args:

        string: The string to be trimmed.

        length: The length to trim the string to, including any ellipses that may be added.

    Returns: The trimmed string, including ellipses if needed.

    """
    if len(string) > length:
        return string[:length - 1] + "…"
    else:
        return string


def has_balanced_parentheses(string: str, left="(", right=")") -> bool:
    """
    Determines whether a string has matching parentheses or not.

    Examples:

        >>> has_balanced_parentheses("3 * (x + (2 ** 5))") -> True

        >>> has_balanced_parentheses("3 * (x + (2 ** 5)") -> False

    Args:

        string: The string to be evaluated.

        left: The left parentheses. Can be modified if, for example, you need to check square brackets.

        right: The right parentheses. Can be modified if, for example, you need to check square brackets.

    Returns: A boolean of whether or not the string has balanced parentheses.

    """
    parenthesis_level = 0

    for char in string:
        if char == left:
            parenthesis_level += 1
        elif char == right:
            parenthesis_level -= 1

    return parenthesis_level == 0


def wrap_text_ignoring_mathtext(
        text: str,
        width: int = 70,
) -> str:
    """
    Reformat the single paragraph in 'text' to fit in lines of no more
    than 'width' columns, and return a new string containing the entire
    wrapped paragraph.  Tabs are expanded and other
    whitespace characters converted to space.

    Similar to `textwrap.fill`, but keeps any mathtext blocks contiguous and unaltered. Mathtext blocks are segments of `text` that are between $ markers, to indicate LaTeX-like formatting. Dollar-sign literals (\$) do not trigger Mathtext, and that is respected here as well.

    For example:
        >>> wrap_text_ignoring_mathtext()

    Args:

        text: The text to be wrapped.

        width: The maximum width of wrapped lines (unless break_long_words is false)

    Returns:

        A string containing the entire paragraph with line breaks as newline ("\n") characters.

    """
    import textwrap, re

    # Pattern to match mathtext blocks
    mathtext_trigger = r"(?<!\\)(?:\\\\)*\$"

    # Split the text into non-mathtext parts and mathtext parts
    parts = re.split(mathtext_trigger, text)
    text_parts = [part for i, part in enumerate(parts) if i % 2 == 0]
    math_parts = [part for i, part in enumerate(parts) if i % 2 == 1]

    # Reassemble th result
    output = ""
    cursor_position = 0

    while len(text_parts) + len(math_parts) > 0:
        try:
            text_part = text_parts.pop(0)

            contribution = textwrap.fill(
                text_part,
                width=width,
                initial_indent=" " * cursor_position,
                drop_whitespace=False,
            )[cursor_position:]

            output += contribution

            if "\n" in contribution:
                cursor_position = len(contribution.split("\n")[-1])
            else:
                cursor_position += len(contribution)

        except IndexError:
            pass

        try:
            math_part = math_parts.pop(0)

            estimated_space: int = int(np.round(len(math_part) * 0.5))

            if cursor_position + estimated_space < width:
                output += f"${math_part}$"
                cursor_position += estimated_space
            else:
                output += f"\n${math_part}$"
                cursor_position = estimated_space

        except IndexError:
            pass

    output = "\n".join([line.strip() for line in output.split("\n")])

    return output


if __name__ == '__main__':
    for input in [
        r"$ax^2+bx+c$",
        r"Photon flux $\phi$",
        r"Photon flux $\phi$ is given by $\phi = \frac{c}{\lambda}$",
        r"Earnings for 2022 $M\$/year$",
        r"$ax^2+bx+c$ and also $3x$"
    ]:
        print(wrap_text_ignoring_mathtext(input, width=10))
