import pytest

from conditions import reduce, hitroll_adv_dis, Conditions, map_reduce
from dice import RollType
import jax.numpy as jnp
import jax
from constants import Abilities


def as_bool_array(condition: Conditions):

    condition_list = jnp.zeros(len(Conditions), jnp.bool)
    return condition_list.at[condition].set(True)


def test_map_reduce():
    condition = map_reduce(as_bool_array(Conditions.POISONED))
    assert condition.hitroll.dtype == jnp.int8
    assert condition.hitroll == RollType.DISADVANTAGE


def test_stack_reduce_poison_exhaustion():
    poisoned = as_bool_array(Conditions.POISONED)
    exhaustion = as_bool_array(Conditions.EXHAUSTION)
    conditions = poisoned | exhaustion
    reduced = map_reduce(conditions)
    assert reduced.hitroll == RollType.DISADVANTAGE
    assert reduced.target_hitroll == RollType.NORMAL
    assert reduced.melee_target_hitroll == RollType.NORMAL
    assert reduced.incapacitated == False
    assert jnp.allclose(reduced.ability_check, RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw[Abilities.WIS], RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw_fail, False)
    assert reduced.melee_target_damage_mul == 1.
    assert jnp.allclose(reduced.damage_resistance, 1.)


def test_stack_reduce_blinded_invisible():
    blinded = as_bool_array(Conditions.BLINDED)
    invisible = as_bool_array(Conditions.INVISIBLE)
    conditions = blinded | invisible
    reduced = map_reduce(conditions)
    assert reduced.hitroll == RollType.NORMAL
    assert reduced.target_hitroll == RollType.NORMAL
    assert reduced.melee_target_hitroll == RollType.NORMAL
    assert reduced.incapacitated == False
    assert jnp.allclose(reduced.ability_check, RollType.NORMAL)
    assert jnp.allclose(reduced.saving_throw, RollType.NORMAL)
    assert jnp.allclose(reduced.saving_throw_fail, False)
    assert reduced.melee_target_damage_mul == 1.
    assert jnp.allclose(reduced.damage_resistance, 1.)


def test_attack_condition_modifiers():
    source = map_reduce(as_bool_array(Conditions.POISONED))
    target = map_reduce(as_bool_array(Conditions.PARALYZED))
    hitroll_type_mod = hitroll_adv_dis(source, target)
    assert hitroll_type_mod == RollType.NORMAL

    source = map_reduce(as_bool_array(Conditions.INVISIBLE))
    target = map_reduce(as_bool_array(Conditions.INVISIBLE))
    hitroll_type_mod = hitroll_adv_dis(source, target)
    assert hitroll_type_mod == RollType.NORMAL

