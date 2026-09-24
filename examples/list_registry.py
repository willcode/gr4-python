"""Lists the block and scheduler types of the registry by family.

    python3 list_registry.py [text]

With text, only the families whose keys contain it, ignoring case, are listed.
"""

import collections
import sys

import gr4
from gr4 import registry


def families(keys: list[str]) -> dict[str, list[str]]:
    """Groups keys by family and returns, for each family, the type parameters of its keys."""
    grouped = collections.defaultdict(list)
    for key in keys:
        parameters = registry.type_parameters_of(key)
        grouped[registry.family_of(key)].append(f"<{', '.join(parameters)}>" if parameters else "")
    return dict(sorted(grouped.items()))


def main() -> None:
    """Prints each family with its type parameters, blocks first and schedulers after."""
    text = sys.argv[1] if len(sys.argv) > 1 else ""
    for title, keys in (("blocks", registry.block_types()), ("schedulers", registry.scheduler_types())):
        matching = [key for key in keys if text.lower() in key.lower()]
        grouped = families(matching)
        print(f"{title}: {len(matching)} types in {len(grouped)} families")
        for family, parameters in grouped.items():
            print(f"  {family} {' '.join(parameters)}".rstrip())
    print(f"default scheduler: {gr4.DEFAULT_SCHEDULER}")


if __name__ == "__main__":
    main()
