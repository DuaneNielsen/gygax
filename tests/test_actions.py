import pytest
from actions import action_table, ActionsEnum, lookup_magic_bonus
import jax


def test_addition():
    longsword = jax.tree.map(lambda x : x[ActionsEnum['longsword']], action_table)
    assert longsword.damage == 4.5
    longsword = longsword + lookup_magic_bonus('+1')
    assert longsword.damage == 5.5