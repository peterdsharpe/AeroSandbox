from datetime import datetime

import numpy as np
from matplotlib.dates import date2num


def ensure_float(value):
    """Make sure datetime values are properly converted to floats."""
    try:
        # the last 3 boolean checks are for arrays with datetime64 and with
        # a timezone, see these SO posts:
        # https://stackoverflow.com/q/60714568/4549682
        # https://stackoverflow.com/q/23063362/4549682
        # somewhere, the datetime64 with timezone is getting converted to 'O' dtype
        if (
                isinstance(value, datetime)
                or isinstance(value, np.datetime64)
                or np.issubdtype(value.dtype, np.datetime64)
                or str(value.dtype).startswith("datetime64")
                or value.dtype == "O"
        ):
            return date2num(value)
        else:  # another numpy dtype like float64
            return value
    except AttributeError:  # possibly int or other float/int dtype
        return value


# From https://www.geeksforgeeks.org/maximum-bipartite-matching/
class GFG:
    def __init__(self, graph):

        # residual graph
        self.graph = graph
        self.ppl = len(graph)
        self.jobs = len(graph[0])

    # A DFS based recursive function that returns true if a matching for vertex
    # u is possible
    def bpm(self, u, matchR, seen):

        # Try every job one by one
        for v in range(self.jobs):

            # If applicant u is interested
            # in job v and v is not seen
            if self.graph[u][v] and not seen[v]:

                # Mark v as visited
                seen[v] = True

                # If job 'v' is not assigned to an applicant OR previously
                # assigned applicant for job v (which is matchR[v]) has an
                # alternate job available. Since v is marked as visited in the
                # above line, matchR[v] in the following recursive call will not
                # get job 'v' again

                if matchR[v] == -1 or self.bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    def maxBPM(self):
        # An array to keep track of the applicants assigned to jobs. The value
        # of matchR[i] is the applicant number assigned to job i, the value -1
        # indicates nobody is assigned.
        matchR = [-1] * self.jobs

        # Count of jobs assigned to applicants
        result = 0
        for i in range(self.ppl):

            # Mark all jobs as not seen for next applicant.
            seen = [False] * self.jobs

            # Find if the applicant 'u' can get a job
            if self.bpm(i, matchR, seen):
                result += 1
        return result, matchR


def maximum_bipartite_matching(graph: np.ndarray) -> np.ndarray:
    """Finds the maximum bipartite matching of a graph

    Parameters
    ----------
    graph : np.ndarray
        The graph, represented as a boolean matrix

    Returns
    -------
    order : np.ndarray
        The order in which to traverse the graph to visit a maximum of nodes
    """
    g = GFG(graph)
    _, order = g.maxBPM()
    return np.asarray(order)


def always_iterable(obj, base_type=(str, bytes)):
    """If *obj* is iterable, return an iterator over its items::

        >>> obj = (1, 2, 3)
        >>> list(always_iterable(obj))
        [1, 2, 3]

    If *obj* is not iterable, return a one-item iterable containing *obj*::

        >>> obj = 1
        >>> list(always_iterable(obj))
        [1]

    If *obj* is ``None``, return an empty iterable:

        >>> obj = None
        >>> list(always_iterable(None))
        []

    By default, binary and text strings are not considered iterable::

        >>> obj = 'foo'
        >>> list(always_iterable(obj))
        ['foo']

    If *base_type* is set, objects for which ``isinstance(obj, base_type)``
    returns ``True`` won't be considered iterable.

        >>> obj = {'a': 1}
        >>> list(always_iterable(obj))  # Iterate over the dict's keys
        ['a']
        >>> list(always_iterable(obj, base_type=dict))  # Treat dicts as a unit
        [{'a': 1}]

    Set *base_type* to ``None`` to avoid any special handling and treat objects
    Python considers iterable as iterable:

        >>> obj = 'foo'
        >>> list(always_iterable(obj, base_type=None))
        ['f', 'o', 'o']
    """
    if obj is None:
        return iter(())

    if (base_type is not None) and isinstance(obj, base_type):
        return iter((obj,))

    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))
