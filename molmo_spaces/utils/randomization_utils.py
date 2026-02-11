import random
from collections.abc import Sequence
from typing import Any


def unzip(seq: Sequence[tuple], n: int | None):
    """Undoes a `zip` operation.

    # Parameters

    seq: The sequence of tuples that should be unzipped
    n: The number of items in each tuple. This is an optional value but is necessary if
       `len(seq) == 0` (as there is no other way to infer how many empty lists were zipped together
        in this case) and can otherwise be used to error check.

    # Returns

    A tuple (of length `n` if `n` is given) of lists where the ith list contains all
    the ith elements from the tuples in the input `seq`.
    """
    assert n is not None or len(seq) != 0
    if n is None:
        n = len(seq[0])
    lists = [[] for _ in range(n)]

    for t in seq:
        assert len(t) == n
        for i in range(n):
            lists[i].append(t[i])
    return lists


def weighted_random_permutation(some_list: Sequence, weights: Sequence[float]) -> list:
    ind_to_weight = {i: w for i, w in enumerate(weights)}

    permuted_inds = []
    for _ in range(len(some_list)):
        subinds, subweights = unzip(list(ind_to_weight.items()), 2)
        permuted_inds.append(
            random.choices(
                subinds,
                weights=subweights,
                k=1,
            )[0]
        )
        del ind_to_weight[permuted_inds[-1]]

    return [some_list[i] for i in permuted_inds]


def weighted_random_permutation_from_counts(
    some_list: Sequence | set, counts: Sequence[float] | dict[Any, float]
) -> list:
    if isinstance(some_list, set):
        assert isinstance(counts, dict), "If l is a set, counts must be a dict."
        some_list = list(some_list)

    if isinstance(counts, dict):
        counts = [counts[ll] for ll in some_list]

    return weighted_random_permutation(
        some_list=some_list, weights=[1.0 / (c + 1e-8) for c in counts]
    )
