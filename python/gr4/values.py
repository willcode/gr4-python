"""Setting values: the coercion of Python values to a setting's type, and the live settings of a block.

A value converts to a pmt value by the extension's rules. Without a type code its Python type selects the value type:
bool to Bool, int to Int64 (UInt64 above the Int64 range), float to Float64, complex to complex<float64>, str to
String, dict to a map, None to the empty value, and a list or tuple to a tensor of type-erased elements. An object
with the buffer protocol converts to the typed tensor of its format and keeps its element type. With a type code the
value converts to exactly that type, and a value that does not fit raises OverflowError (out of range, or a fraction
for an integer type) or TypeError (the wrong kind), naming the key and the type it needs.

The package passes a setting's type code wherever it knows one. The block's setting types give it when a value is
staged and when a payload goes to the block's Settings or StagedSettings endpoint. The type's description gives it when
a block is added, when a scheduler is created with settings, and when settings are merged into a graph document. An
integer a float32 cannot hold exactly therefore reaches a float32 setting rounded, and a complex reaches a
complex<float32> one; the settings code refuses both without the code.
"""

import difflib
from collections.abc import Iterable, Iterator, Mapping

from . import _gr4
from .errors import SettingError, translated

convert = _gr4.convert
type_code = _gr4.type_code

UNTYPED_CODES = frozenset({"none", "tensor<value>"})
"""The type codes coercion leaves out: none admits only None, and tensor<value> only a list or tuple."""


def coercion_types(declared: Mapping[str, str], keys: Iterable[str]) -> dict[str, str]:
    """Returns the declared type code of each of keys, for the extension's typed conversion.

    A key without a declared code, or with a code in UNTYPED_CODES, is left out and takes the type its Python value
    selects.
    """
    return {key: declared[key] for key in keys if key in declared and declared[key] not in UNTYPED_CODES}


def closest(key: str, candidates: Iterable[str]) -> list[str]:
    """Returns up to three of candidates that are spelled like key, the closest first."""
    return difflib.get_close_matches(key, list(candidates), n=3, cutoff=0.6)


def unknown_setting(key: str, declared: Iterable[str], owner: str) -> KeyError:
    """Returns the KeyError for a setting owner does not declare, naming the closest declared keys."""
    suggestions = closest(key, declared)
    hint = f"; did you mean {', '.join(repr(name) for name in suggestions)}?" if suggestions else ""
    return KeyError(f"{owner} has no setting '{key}'{hint}")


class Settings(Mapping[str, object]):
    """The settings of one block, read live.

    settings[key] reads the value in force. settings[key] = value and update() stage values, converted to each
    setting's type, and the block applies them at its next work call. dict(settings) is a snapshot of the values in
    force; defaults, staged and types are the other maps.
    """

    def __init__(self, handle: _gr4.BlockHandle, owner: str) -> None:
        """Views the settings of the block handle names; owner names the block in error messages."""
        self._handle = handle
        self._owner = owner

    def __getitem__(self, key: str) -> object:
        """Returns the value in force. KeyError, naming the closest settings, when the block has no such setting."""
        try:
            return self._handle.get_value(key)
        except KeyError:
            raise unknown_setting(key, self._handle.setting_types(), self._owner) from None

    def __iter__(self) -> Iterator[str]:
        """Iterates over the keys of the settings."""
        return iter(self._handle.setting_types())

    def __len__(self) -> int:
        """Returns the number of settings."""
        return len(self._handle.setting_types())

    def __contains__(self, key: object) -> bool:
        """True when the block has a setting named key."""
        return key in self._handle.setting_types()

    def __setitem__(self, key: str, value: object) -> None:
        """Stages one value; see update()."""
        self.update({key: value})

    def update(self, values: Mapping[str, object] | Iterable[tuple[str, object]] = (), /, **settings: object) -> None:
        """Stages values for the block's next work call, each converted to its setting's type.

        A key the block does not declare raises KeyError naming the closest keys, and nothing is staged. A value of
        the wrong kind raises TypeError, and one out of range OverflowError. A value the block refuses raises
        SettingError.
        """
        staged = dict(values, **settings)
        declared = self._handle.setting_types()
        for key in staged:
            if key not in declared:
                raise unknown_setting(key, declared, self._owner)
        with translated(SettingError):
            refused = self._handle.set_staged(staged, coercion_types(declared, staged))
        if refused:
            raise SettingError(f"{self._owner} did not stage {', '.join(sorted(refused))}", "Settings.update")

    @property
    def defaults(self) -> dict[str, object]:
        """The values a reset to defaults restores."""
        return self._handle.default_parameters()

    @property
    def staged(self) -> dict[str, object]:
        """The values staged for the block's next work call."""
        return self._handle.staged_parameters()

    @property
    def types(self) -> dict[str, str]:
        """The type code of each setting, by key."""
        return self._handle.setting_types()

    def __repr__(self) -> str:
        """Names the block and lists the values in force."""
        return f"<Settings of {self._owner}: {self._handle.get()!r}>"
