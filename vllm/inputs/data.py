from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)
from typing_extensions import NotRequired, TypedDict, TypeVar, assert_never


if TYPE_CHECKING:
    from vllm.multimodal import (
        MultiModalDataDict,
        MultiModalKwargs,
        MultiModalPlaceholderDict,
    )
    from vllm.multimodal.inputs import MultiModalInputsV2


class TokenInputs(TypedDict):
    """Represents token-based inputs."""

    type: Literal["token"]
    """The type of inputs."""

    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    token_type_ids: NotRequired[List[int]]
    """The token type IDs of the prompt."""

    prompt: NotRequired[str]
    """
    The original prompt text corresponding to the token IDs, if available.
    """

    multi_modal_data: NotRequired["MultiModalDataDict"]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    multi_modal_inputs: NotRequired["MultiModalKwargs"]
    """
    Optional multi-modal inputs to pass to the model,
    if the model supports it.
    """

    multi_modal_placeholders: NotRequired["MultiModalPlaceholderDict"]
    """
    Placeholder ranges for the multi-modal data.
    """

    mm_processor_kwargs: NotRequired[Dict[str, Any]]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor. Note that if multiple modalities
    have registered mappers etc for the model being considered, we attempt
    to pass the mm_processor_kwargs to each of them.
    """


DecoderOnlyInputs = Union[TokenInputs, "MultiModalInputsV2"]
"""
The inputs in :class:`~vllm.LLMEngine` before they are
passed to the model executor.
This specifies the data required for decoder-only models.
"""
