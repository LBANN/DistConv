import inspect

import torch


def signature_from_schema(op: torch._ops.OpOverload) -> inspect.Signature:
    """
    Build a Python signature from an aten op's schema. This is useful for binding the arguments
    of builtin torch functions, which do not have an inspectable signature.

    Note: Default values in the schema are not included in the signature.

    Args:
        op (torch._ops.OpOverload): The aten op overload, e.g. torch.ops.aten.instance_norm.default.

    Returns:
        inspect.Signature: The signature of the op.
    """
    return inspect.Signature(
        [
            inspect.Parameter(
                arg.name,
                inspect.Parameter.KEYWORD_ONLY
                if arg.kwarg_only
                else inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for arg in op._schema.arguments
        ]
    )
