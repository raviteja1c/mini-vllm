from dataclasses import dataclass
from functools import cached_property
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

import torch
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


def token_inputs(
    prompt_token_ids: List[int],
    token_type_ids: Optional[List[int]] = None,
    prompt: Optional[str] = None,
    multi_modal_data: Optional["MultiModalDataDict"] = None,
    multi_modal_inputs: Optional["MultiModalKwargs"] = None,
    multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None,
    mm_processor_kwargs: Optional[Dict[str, Any]] = None,
) -> TokenInputs:
    """Construct :class:`TokenInputs` from optional values."""
    inputs = TokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if token_type_ids is not None:
        inputs["token_type_ids"] = token_type_ids
    if multi_modal_data is not None:
        inputs["multi_modal_data"] = multi_modal_data
    if multi_modal_inputs is not None:
        inputs["multi_modal_inputs"] = multi_modal_inputs
    if multi_modal_placeholders is not None:
        inputs["multi_modal_placeholders"] = multi_modal_placeholders
    if mm_processor_kwargs is not None:
        inputs["mm_processor_kwargs"] = mm_processor_kwargs

    return inputs


DecoderOnlyInputs = Union[TokenInputs, "MultiModalInputsV2"]
"""
The inputs in :class:`~vllm.LLMEngine` before they are
passed to the model executor.
This specifies the data required for decoder-only models.
"""


SingletonInputs = Union[TokenInputs, "MultiModalInputsV2"]
"""
A processed :class:`SingletonPrompt` which can be passed to
:class:`vllm.sequence.Sequence`.
"""


@dataclass
class SingletonInputsAdapter:
    """
    Unified interface to access the components of :class:`SingletonInputs`.
    """

    inputs: SingletonInputs

    @cached_property
    def prompt(self) -> Optional[str]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt")

        assert_never(inputs)

    @cached_property
    def prompt_token_ids(self) -> List[int]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("prompt_token_ids", [])

        assert_never(inputs)

    @cached_property
    def token_type_ids(self) -> List[int]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return inputs.get("token_type_ids", [])

        assert_never(inputs)

    @cached_property
    def prompt_embeds(self) -> Optional[torch.Tensor]:
        inputs = self.inputs

        if inputs["type"] == "token" or inputs["type"] == "multimodal":
            return None

        assert_never(inputs)

    @cached_property
    def multi_modal_data(self) -> "MultiModalDataDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_data", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_kwargs", {})

        assert_never(inputs)

    @cached_property
    def multi_modal_inputs(self) -> Union[Dict, "MultiModalKwargs"]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_inputs", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_kwargs", {})

        assert_never(inputs)

    @cached_property
    def multi_modal_placeholders(self) -> "MultiModalPlaceholderDict":
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("multi_modal_placeholders", {})

        if inputs["type"] == "multimodal":
            return inputs.get("mm_placeholders", {})

        assert_never(inputs)

    @cached_property
    def mm_processor_kwargs(self) -> Dict[str, Any]:
        inputs = self.inputs

        if inputs["type"] == "token":
            return inputs.get("mm_processor_kwargs", {})

        if inputs["type"] == "multimodal":
            return {}

        assert_never(inputs)
