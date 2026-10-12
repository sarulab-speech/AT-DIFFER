"""Reading the shared resources, and mapping difference types to output types.

The metadata uses the difference types of the paper (``difference_types`` in
resources/taxonomy.yaml), while the model writes the type names fixed by the
prompt (``output_types``) on its TYPES line. A semantic difference has no
output type of its own: it is reported as a text omission plus a sound
omission (``event_omission`` + ``phantom_sound``), and the prompt forbids any
other name for it. Build every TYPES line through :meth:`Taxonomy.to_output_types`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

# Defaults, relative to the stage directory (finetune/).
DEFAULT_TAXONOMY_FILE = "../resources/taxonomy.yaml"
DEFAULT_OUTPUT_FORMAT_FILE = "../resources/output_format.yaml"

#: Difference type of an item whose text describes its audio correctly.
NO_DIFFERENCE = "no_difference"


@dataclass(frozen=True)
class Taxonomy:
    """The definitions in resources/ that are needed to build the targets."""

    #: Type names allowed on the TYPES line, in the order they appear there.
    output_types: tuple[str, ...]
    #: Each difference type expressed as output types.
    difference_type_to_output_types: Mapping[str, tuple[str, ...]]
    #: Line keys in output order: DISCREPANCY, TYPES, EXPLANATION.
    keys: tuple[str, str, str]
    #: DISCREPANCY value when the text differs from the audio, and when it does not.
    discrepancy_with_difference: str
    discrepancy_no_difference: str
    #: Separator between output types on the TYPES line, and the value used
    #: when there are none.
    types_separator: str
    types_none: str
    #: EXPLANATION value when there is no difference.
    no_difference_explanation: str

    @classmethod
    def load(
        cls,
        taxonomy_file: str | Path = DEFAULT_TAXONOMY_FILE,
        output_format_file: str | Path = DEFAULT_OUTPUT_FORMAT_FILE,
    ) -> "Taxonomy":
        taxonomy = _read_yaml(taxonomy_file)
        output_format = _read_yaml(output_format_file)
        keys = tuple(output_format["keys"])
        if len(keys) != 3:
            raise ValueError(f"{output_format_file}: expected 3 keys, got {list(keys)}")
        return cls(
            output_types=tuple(taxonomy["output_types"]),
            difference_type_to_output_types={
                name: tuple(types)
                for name, types in taxonomy["difference_type_to_output_types"].items()
            },
            keys=keys,
            discrepancy_with_difference=str(output_format["discrepancy"]["with_difference"]),
            discrepancy_no_difference=str(output_format["discrepancy"]["no_difference"]),
            types_separator=output_format["types"]["separator"],
            types_none=output_format["types"]["none"],
            no_difference_explanation=output_format["no_difference_explanation"],
        )

    def to_output_types(self, difference_types: list[str]) -> list[str]:
        """Map difference types to the output types of the TYPES line.

        The result is deduplicated and sorted in the order of ``output_types``:
        the TYPES line says which types apply, not how many times, so two
        semantic differences still give a single event_omission, phantom_sound
        pair.

        >>> taxonomy.to_output_types(["semantic_difference", "simultaneity_discrepancy"])
        ['event_omission', 'phantom_sound', 'overlap']
        >>> taxonomy.to_output_types(["no_difference"])
        []
        """
        seen: set[str] = set()
        for raw in difference_types:
            key = raw.strip().lower()
            if not key:
                continue
            try:
                expanded = self.difference_type_to_output_types[key]
            except KeyError:
                raise ValueError(
                    f"Unknown difference type {raw!r}. Known types: "
                    f"{sorted(self.difference_type_to_output_types)}."
                ) from None
            seen.update(expanded)
        return [name for name in self.output_types if name in seen]

    def format_types_line(self, output_types: list[str]) -> str:
        """The value of the TYPES line (``types.none`` if there are no types)."""
        return self.types_separator.join(output_types) if output_types else self.types_none


def _read_yaml(path: str | Path) -> dict:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Resource file not found: {resolved}")
    return yaml.safe_load(resolved.read_text(encoding="utf-8"))
