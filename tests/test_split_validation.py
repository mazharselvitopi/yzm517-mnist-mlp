"""Boyut sözleşmesi: sessiz kesme/TypeError yerine anlaşılır ValueError."""
import pytest
from core import make_split


@pytest.mark.parametrize('args', [
    (10, 2.5, 42, None, None),
    (10.5, 2, 42, None, None),
    (10, 2, 42, 2.5, None),
    (10, 2, 42, None, 1.5),
    (10, True, 42, None, None),
    (10, 2, -1, None, None),
])
def test_split_rejects_non_integer_or_invalid_inputs(args):
    with pytest.raises(ValueError):
        make_split(*args)
