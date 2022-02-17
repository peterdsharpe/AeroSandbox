import inspect
from typing import List
from pathlib import Path


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

    filename, lineno = get_caller_source_location(stacklevel=stacklevel + 1)

    ### Initialize the caller source lines, which is a list of strings that contain the source for the call.
    caller_source_lines: List[str] = []

    ### Read the source lines of code surrounding the call
    try:
        with open(filename, "r") as f:  # Read the file containing the call
            for _ in range(lineno - 1):  # Skip the first N lines of code, until you get to the call
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
            "Couldn't retrieve source code at this stack level, because the source code file couldn't be opened for some reason.")

    caller_source = "\n".join(caller_source_lines)

    return caller_source


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
