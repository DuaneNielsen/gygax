from conditions import CONDITIONS, ConditionModArray, reduce_condition_mod_arrays
from character import convert
from constants import Conditions
from dice import RollType
import jax.numpy as jnp
import jax
from jax.tree_util import tree_reduce

def test_conversion():
    condition = CONDITIONS[Conditions.POISONED]
    assert condition.condition == Conditions.POISONED
    condition_mod = convert(condition, ConditionModArray)
    assert condition_mod.hitroll.dtype == jnp.int8
    assert condition_mod.hitroll == RollType.DISADVANTAGE


def test_reduce():
    poisoned = convert(CONDITIONS[Conditions.POISONED], ConditionModArray)
    exhaustion = convert(CONDITIONS[Conditions.EXHAUSTION], ConditionModArray)
    condition_stack = jax.tree.map(lambda *x : jnp.stack(x), *[poisoned, exhaustion])
    reduced = reduce_condition_mod_arrays(condition_stack)
    assert reduced.hitroll == RollType.DISADVANTAGE
    assert reduced.target_hitroll == RollType.NORMAL
    assert reduced.melee_hitroll == RollType.NORMAL
    assert reduced.incapacitated == False
    assert jnp.allclose(reduced.ability_check, RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw, RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw_fail, False)
    assert reduced.melee_damage_multiplier == 1.
    assert jnp.allclose(reduced.damage_resistance, 1.)

