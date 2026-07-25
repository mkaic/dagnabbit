"""Automatic Muon/AdamW parameter routing on top of the built-in optimizers.

Muon (Jordan et al., 2024) orthogonalizes a weight matrix's momentum buffer
before applying it, which only means something for parameters that actually
*are* linear maps between two hidden feature spaces: the fully-connected
weights inside transformer blocks (attention projections and MLPs). It is not
meant for 1-D parameters (biases, LayerNorm gains), for lookup tables
(embeddings, learned position/register/mask tokens), or for the final
classifier head, whose output axis indexes classes rather than a hidden space.

``torch.optim.Muon`` implements the update rule but leaves the routing to the
caller, and its own docstring suggests splitting on ``p.ndim == 2``. That
heuristic is wrong for this model: ``nn.Embedding`` tables and the learned
position/register token banks are all 2-D but are not hidden weight matrices.
:class:`AutoMuon` does the split by the type of the module that owns each
parameter instead, and drives ``torch.optim.Muon`` and ``torch.optim.AdamW``
underneath as one optimizer so checkpointing and LR scheduling stay uniform.
"""

from fnmatch import fnmatch
from typing import Iterable, Sequence

import torch
import torch.nn as nn

# Module types whose 2-D weights are genuine hidden-to-hidden linear maps.
# Anything else routes to AdamW, so an unrecognized module type degrades to the
# safe option rather than to a silently wrong update rule.
MUON_MODULE_TYPES = (
    nn.Linear,
    nn.LazyLinear,
    nn.MultiheadAttention,
)


def build_optimizer(
    optimizer_class: type[torch.optim.Optimizer],
    model: nn.Module,
    **kwargs,
) -> torch.optim.Optimizer:
    """Construct ``optimizer_class`` for ``model``.

    :class:`AutoMuon` needs the module tree to route parameters; every other
    ``Optimizer`` takes the usual flat parameter iterable.
    """
    if issubclass(optimizer_class, AutoMuon):
        return optimizer_class(model, **kwargs)
    return optimizer_class(model.parameters(), **kwargs)


class AutoMuon(torch.optim.Optimizer):
    """``torch.optim.Muon`` on hidden weight matrices, ``AdamW`` on the rest.

    ``model`` is walked once at construction and each trainable parameter is
    routed by the type of the module that *owns* it:

    * 2-D weights of :data:`MUON_MODULE_TYPES` -> Muon.
    * everything else -> AdamW. That covers biases, LayerNorm gains,
      ``nn.Embedding`` tables, and bare ``nn.Parameter`` token banks such as
      learned position/register/mask embeddings -- all things Muon is not for,
      several of which a plain ``ndim == 2`` split would misroute.

    Output heads are the one case module type cannot settle: a classifier is
    just an ``nn.Linear``, but its output axis indexes classes rather than a
    hidden space, so it belongs on AdamW. Name them in ``adam_module_names``
    (each entry matches a parameter path exactly, matches a whole submodule
    subtree by prefix, or is an fnmatch pattern).

    The two rules get separate ``lr`` values in separate param groups. With the
    default ``adjust_lr_fn="original"`` Muon wants a much larger LR than AdamW
    (~0.02 against ~3e-4); with ``"match_rms_adamw"`` it is scaled to reuse an
    AdamW-tuned LR directly. Either way an ``LRScheduler`` multiplies both
    groups by the same factor, so warmup and decay behave as usual.

    Caveat on fused QKV: ``nn.MultiheadAttention`` packs Q, K and V into one
    ``[3 * embed_dim, embed_dim]`` ``in_proj_weight``. Muon orthogonalizes it
    as a single matrix, which is not identical to orthogonalizing the three
    projections separately (implementations built around Muon usually keep them
    as separate parameters). The LR adjustment still normalizes the update's
    scale, so this is a fidelity caveat rather than a correctness one; pass
    ``adam_module_names=("*.in_proj_weight",)`` to keep fused blocks on AdamW.

    Single-device only: this does no distributed gradient sharding.
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        adam_lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
        muon_weight_decay: float = 0.1,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_weight_decay: float = 0.01,
        adam_module_names: Sequence[str] = (),
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(
                "AutoMuon classifies parameters by the module that owns them, so "
                "it needs the nn.Module itself, not model.parameters()"
            )

        self.adam_module_names = tuple(adam_module_names)
        muon_params, adam_params = self._classify_parameters(model)
        self.muon_parameter_names = [name for name, _ in muon_params]
        self.adam_parameter_names = [name for name, _ in adam_params]

        self.muon = (
            torch.optim.Muon(
                [param for _, param in muon_params],
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                adjust_lr_fn=adjust_lr_fn,
            )
            if muon_params
            else None
        )
        self.adamw = (
            torch.optim.AdamW(
                [param for _, param in adam_params],
                lr=adam_lr,
                betas=adam_betas,
                eps=adam_eps,
                weight_decay=adam_weight_decay,
            )
            if adam_params
            else None
        )
        if self.muon is None and self.adamw is None:
            raise ValueError("model has no trainable parameters")

        # Adopt the children's own group dicts rather than copies: an
        # LRScheduler mutating ``group["lr"]`` here is then the same mutation
        # the child reads on its next step.
        super().__init__(self._child_param_groups(), {})

    @property
    def _children(self) -> list[torch.optim.Optimizer]:
        return [child for child in (self.muon, self.adamw) if child is not None]

    def _child_param_groups(self) -> list[dict]:
        return [group for child in self._children for group in child.param_groups]

    # ---- parameter routing ----

    def _is_forced_to_adam(self, parameter_path: str) -> bool:
        for pattern in self.adam_module_names:
            if (
                parameter_path == pattern
                or parameter_path.startswith(pattern + ".")
                or fnmatch(parameter_path, pattern)
            ):
                return True
        return False

    def _classify_parameters(
        self,
        model: nn.Module,
    ) -> tuple[list[tuple[str, nn.Parameter]], list[tuple[str, nn.Parameter]]]:
        muon_params: list[tuple[str, nn.Parameter]] = []
        adam_params: list[tuple[str, nn.Parameter]] = []
        seen_parameter_ids: set[int] = set()

        for module_name, module in model.named_modules():
            is_muon_module = isinstance(module, MUON_MODULE_TYPES)
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                # Shared/tied parameters surface under several names; the first
                # one wins so no parameter lands in two groups.
                if id(param) in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(id(param))

                parameter_path = (
                    f"{module_name}.{param_name}" if module_name else param_name
                )
                # torch.optim.Muon accepts 2-D parameters only, which also
                # excludes MultiheadAttention's [1, 1, E] bias_k/bias_v.
                use_muon = (
                    is_muon_module
                    and param.ndim == 2
                    and "bias" not in param_name
                    and not self._is_forced_to_adam(parameter_path)
                )
                if use_muon:
                    muon_params.append((parameter_path, param))
                else:
                    adam_params.append((parameter_path, param))

        return muon_params, adam_params

    def summary(self) -> str:
        """One-line report of how parameters were routed."""

        def describe(params: Iterable[nn.Parameter]) -> str:
            params = list(params)
            total = sum(param.numel() for param in params)
            return f"{len(params)} tensors, {total} elements"

        def params_of(child: torch.optim.Optimizer | None) -> list[nn.Parameter]:
            if child is None:
                return []
            return [param for group in child.param_groups for param in group["params"]]

        return (
            f"AutoMuon: muon={describe(params_of(self.muon))} "
            f"adamw={describe(params_of(self.adamw))}"
        )

    # ---- optimizer interface ----

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for child in self._children:
            child.step()
        return loss

    def state_dict(self) -> dict:
        """The two children's state dicts, keyed by rule.

        Not interchangeable with a plain ``Optimizer`` state dict; keeping them
        separate avoids having to reconcile two independent parameter indexes.
        """
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        for key, child in (("muon", self.muon), ("adamw", self.adamw)):
            if (child is None) != (state_dict[key] is None):
                raise ValueError(
                    f"checkpoint {'has' if child is None else 'is missing'} a "
                    f"'{key}' optimizer state that this AutoMuon does not match"
                )
            if child is not None:
                child.load_state_dict(state_dict[key])
        # A child's load_state_dict replaces its group dicts with fresh ones,
        # which would leave this optimizer (and any attached LRScheduler)
        # holding the orphaned originals.
        self.param_groups = self._child_param_groups()
