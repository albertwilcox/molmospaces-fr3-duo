import concurrent.futures
from collections.abc import Sequence
from typing import Any


def wait_for_futures_and_raise_errors(
    futures: Sequence[concurrent.futures.Future],
) -> Sequence[Any]:
    results = []
    concurrent.futures.wait(futures)
    for future in futures:
        try:
            results.append(future.result())  # This will re-raise any exceptions
        except Exception:
            raise
    return results
