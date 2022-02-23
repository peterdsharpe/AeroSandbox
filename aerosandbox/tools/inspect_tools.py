import inspect
from typing import List
from pathlib import Path
from aerosandbox.tools.string_formatting import has_balanced_parentheses


def get_caller_source_location(
        stacklevel: int = 1,
) -> (Path, int):
    ### Go up `stacklevel` frames from the current one to get to the caller frame.
    frame = inspect.currentframe()
    for _ in range(stacklevel):
        frame = frame.f_back

    ### Extract the frame info (an `inspect.Traceback` type) from the caller frame
    frame_info: inspect.Traceback = inspect.getframeinfo(frame)

    filename = Path(frame_info.filename)
    lineno = frame_info.lineno

    return filename, lineno


def get_caller_source_code(
        stacklevel: int = 1,
        strip_lines: bool = False
) -> str:
    """
    Gets the source code of wherever this function is called.

    You can get source code from further up the call stack by using the `stacklevel` argument.

    Args:

        stacklevel: Choose the level of the stack that you want to retrieve source code at. Higher integers will get
        you higher (i.e., more end-user-facing) in the stack. Same behaviour as the `stacklevel` argument in
        warnings.warn().

        strip_lines: A boolean flag about whether or not to strip leading and trailing whitespace off each line of a
        multi-line function call. See the built-in string method `str.strip()` for behaviour.

    Returns: The source code of the call, as a string. Might be a multi-line string (i.e., contains '\n' characters)
    if the call is multi-line.

    """
    ### Go up `stacklevel` frames from the current one to get to the caller frame.
    frame = inspect.currentframe()
    for _ in range(stacklevel):
        frame = frame.f_back

    ### Extract the frame info (an `inspect.Traceback` type) from the caller frame
    frame_info: inspect.Traceback = inspect.getframeinfo(frame)

    ### If Python's auto-extracted "code context" is a compete statement, then you're done here.
    code_context = "".join(frame_info.code_context)
    if has_balanced_parentheses(code_context):
        return code_context

    ### Initialize the caller source lines, which is a list of strings that contain the source for the call.
    caller_source_lines: List[str] = []

    ### Read the source lines of code surrounding the call
    try:
        with open(frame_info.filename, "r") as f:  # Read the file containing the call
            for _ in range(frame_info.lineno - 1):  # Skip the first N lines of code, until you get to the call
                f.readline()  # Unfortunately there's no way around this, since you need to find the "\n" encodings in the file

            parenthesis_level = 0  # Track the number of "(" and ")" characters, so you know when the function call is complete

            def add_line() -> None:
                """
                Adds the subsequent line to the caller source lines (`caller_source_lines`). In-place.
                """
                line = f.readline()

                if strip_lines:
                    line = line.strip()

                nonlocal parenthesis_level
                for char in line:
                    if char == "(":
                        parenthesis_level += 1
                    elif char == ")":
                        parenthesis_level -= 1
                caller_source_lines.append(line)

            ### Get the first line, which is always part of the function call, and includes the opening parenthesis
            add_line()

            ### Do subsequent lines
            while parenthesis_level > 0:
                add_line()
    except OSError as e:
        raise FileNotFoundError(
            "\n".join([
                "Couldn't retrieve source code at this stack level, because the source code file couldn't be opened for some reason.",
                "One common possible reason is that you're referring to an IPython console with a multi-line statement."
            ])
        )

    caller_source = "\n".join(caller_source_lines)

    return caller_source


def get_function_argument_names_from_source_code(source_code: str) -> List[str]:
    """
    Gets the names of the function arguments found in a particular line of source code.

    Specifically, it retrieves the names of the arguments in the first function call found in the source code string.

    If the source code line is an assignment statement, only the right-hand-side of the line is analyzed.

    Also, removes all line breaks ('\n').

    Examples function inputs and outputs:

        "f(a, b)"                -> ['a', 'b']
        "f(a,b)"                 -> ['a', 'b']
        "f(\na,\nb)"             -> ['a', 'b']
        "g = f(a, b)"            -> ['a', 'b']
        "g.h = f(a, b)"          -> ['a', 'b']
        "g.h() = f(a, b)"        -> ['a', 'b']
        "g.h(i=j) = f(a, b)"     -> ['a', 'b']
        "f(a, b) + g(h)"         -> ['a', 'b']
        "f(a: int, b: MyType())" -> ['a', 'b']
        "f(a, b).g(c, d)"        -> ['a', 'b']
        "f(a(b), c)"             -> ['a(b)', 'c']
        "f(a(b,c), d)"           -> ['a(b,c)', 'd']
        "f({a:b}, c)"            -> ['{a:b}', 'c']
        "f(a[b], c)"             -> ['a[b]', 'c']
        "f({a:b, c:d}, e)"       -> ['{a:b,c:d}', 'e']
        "f({a:b,\nc:d}, e)"      -> ['{a:b,c:d}', 'e']
        "f(dict(a=b,c=d), e)"    -> ['dict(a=b,c=d)', 'e']
        "f(a=1, b=2)"            -> ['a=1', 'b=2']
        "f()"                    -> ['']
        "f(incomplete, "         -> raises ValueError
        "3 + 5"                  -> raises ValueError
        ""                       -> raises ValueError

    Args:
        source_code: A line of Python source code that includes a function call. Can be a multi-line piece of source code (e.g., includes '\n').

    Returns: A list of strings containing all of the function arguments. If keyword arguments are found, includes both the key and the value, as-written.

    """

    assignment_equals_index = 0

    parenthesis_level = 0
    for i, char in enumerate(source_code):
        if char == "(":
            parenthesis_level += 1
        elif char == ")":
            parenthesis_level -= 1
        elif char == "=" and parenthesis_level == 0:
            assignment_equals_index = i + 1
            break

    source_code_rhs = source_code[assignment_equals_index:]

    source_code_rhs = source_code_rhs.replace("\n", "")

    parenthesis_level = 0
    braces_level = 0
    for i, char in enumerate(source_code_rhs):
        if char == "(":
            parenthesis_level += 1
            break

    if parenthesis_level == 0:
        raise ValueError("No function call was found in the source code provided!")

    arg_names: List[str] = []
    current_arg = ""
    in_type_hinting_block = False

    while parenthesis_level != 0:
        i += 1
        if i >= len(source_code_rhs):
            raise ValueError("Couldn't match all parentheses, so this doesn't look like valid code!")
        char = source_code_rhs[i]

        if char == "(":
            parenthesis_level += 1
        elif char == ")":
            parenthesis_level -= 1
        elif char == "{":
            braces_level += 1
        elif char == "}":
            braces_level -= 1

        if char == "," and parenthesis_level == 1 and braces_level == 0:
            arg_names.append(current_arg)
            current_arg = ""
            in_type_hinting_block = False
        elif char == ":" and parenthesis_level == 1 and braces_level == 0:
            in_type_hinting_block = True
        elif char == " ":
            pass
        elif parenthesis_level >= 1 and not in_type_hinting_block:
            current_arg += char

    arg_names.append(current_arg)

    def clean(s: str) -> str:
        return s.strip()

    arg_names = [
        clean(arg) for arg in arg_names
    ]

    return arg_names


if __name__ == '__main__':
    def dashes():
        """A quick macro for drawing some dashes, to make the terminal output clearer to distinguish."""
        print("\n" + "-" * 50 + "\n")


    dashes()

    print("Caller location:\n", get_caller_source_location(stacklevel=1))

    dashes()

    print("Caller source code:\n", get_caller_source_code(stacklevel=1))

    dashes()


    def my_func():
        print(
            get_caller_source_code(
                stacklevel=2
            )
        )


    print("Caller source code of a function call:")

    if_you_can_see_this_it_works = my_func()

    dashes()

    print("Arguments of f(a, b):")

    print(
        get_function_argument_names_from_source_code("f(a, b)")
    )
