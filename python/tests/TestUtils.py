# Helper utilities for the assignment tests

from typing import List, Union


DefaultTolerance: float = 0.1


def isClose(a: Union[int, float], b: Union[int, float], tolerance: float) -> bool:
    if isinstance(a, int):
        if isinstance(b, int):
            return a == b
        return False  # the types of a and b did not match
    if isinstance(a, float):
        if isinstance(b, float):
            if a == 0.0:
                return abs(b) < tolerance
            return abs(b - a) / abs(a) < tolerance
        return False  # the types of a and b did not match
    return False  # unsupported type for a


def sortByFirstElement(array: List[tuple]) -> List[tuple]:
    return sorted(array, key=lambda x: x[0])


def checkArray(array1: List[tuple], array2: List[tuple], tolerance: float) -> bool:
    if not isinstance(array1, list) or not isinstance(array2, list) or len(array1) != len(array2):
        return False

    for element1, element2 in zip(sortByFirstElement(array1), sortByFirstElement(array2)):
        if not isinstance(element1, tuple) or not isinstance(element2, tuple) or len(element1) != len(element2):
            return False
        for value1, value2 in zip(element1, element2):
            if not isClose(value1, value2, tolerance):
                return False
    return True


def checkArrays(
    inputArray: List[tuple],
    referenceArray1: List[tuple],
    referenceArray2: List[tuple],
    tolerance: float
) -> bool:
    return (
        checkArray(inputArray, referenceArray1, tolerance) or
        checkArray(inputArray, referenceArray2, tolerance)
    )


def getErrorMessage(inputArray: List[tuple]) -> str:
    return f"{toString(inputArray)} did match the reference values"


def toString(array: List[tuple]) -> str:
    if array is None:
        return str(array)

    return ", ".join(
        list(
            map(
                lambda valueTuple: str(
                    tuple(
                        map(
                            lambda value: round(value, 3) if isinstance(value, (int, float)) else str(value),
                            valueTuple if isinstance(valueTuple, tuple) else (valueTuple,)
                        )
                    )
                ),
                array if isinstance(array, list) else [array]
            )
        )
    )
