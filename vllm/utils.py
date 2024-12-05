from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Literal,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
from typing_extensions import ParamSpec, TypeIs, assert_never

T = TypeVar("T")
U = TypeVar("U")


# `collections` helpers
def is_list_of(
    value: object,
    typ: Type[T],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[List[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


JSONTree = Union[
    Dict[str, "JSONTree[T]"], List["JSONTree[T]"], Tuple["JSONTree[T]", ...], T
]
"""A nested JSON structure where the leaves need not be JSON-serializable."""


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: Dict[str, JSONTree[T]],
) -> Dict[str, JSONTree[U]]: ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: List[JSONTree[T]],
) -> List[JSONTree[U]]: ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: Tuple[JSONTree[T], ...],
) -> Tuple[JSONTree[U], ...]: ...


@overload
def json_map_leaves(
    func: Callable[[T], U],
    value: JSONTree[T],
) -> JSONTree[U]: ...


def json_map_leaves(func: Callable[[T], U], value: JSONTree[T]) -> JSONTree[U]:
    if isinstance(value, dict):
        return {k: json_map_leaves(func, v) for k, v in value.items()}
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)
