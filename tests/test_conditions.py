import pytest

from conditions import reduce, hitroll_adv_dis, Conditions, map_reduce, ConditionStateArray, ConditionState
from dice import RollType
import jax.numpy as jnp
import jax
from constants import Abilities


def as_bool_array(condition: Conditions):

    condition_list = jnp.zeros(len(Conditions), jnp.bool)
    return condition_list.at[condition].set(True)


def test_conditions_add():

    poisoned = ConditionState(poisoned=True).jax()
    incap = ConditionState(incapacitated=True, invisible=True).jax()
    poisoned_and_incap = poisoned + incap
    assert poisoned_and_incap.value[Conditions.POISONED]
    assert poisoned_and_incap.value[Conditions.INCAPACITATED]
    assert poisoned_and_incap.value[Conditions.INVISIBLE]
    not_invis = poisoned_and_incap + ConditionState(invisible=False).jax()
    assert not_invis.value[Conditions.POISONED]
    assert not_invis.value[Conditions.INCAPACITATED]
    assert not not_invis.value[Conditions.INVISIBLE]
    no_change = not_invis + ConditionState().jax()
    assert no_change.value[Conditions.POISONED]
    assert no_change.value[Conditions.INCAPACITATED]
    assert not no_change.value[Conditions.INVISIBLE]
    no_change = no_change + no_change
    assert no_change.value[Conditions.POISONED]
    assert no_change.value[Conditions.INCAPACITATED]
    assert not no_change.value[Conditions.INVISIBLE]
    no_change = no_change + ConditionState(poisoned=True).jax()
    assert no_change.value[Conditions.POISONED]
    assert no_change.value[Conditions.INCAPACITATED]
    assert not no_change.value[Conditions.INVISIBLE]

    # left
    state = ConditionState().jax()
    state = state + ConditionState(poisoned=True).jax() + ConditionState(invisible=True).jax()
    assert state.value[Conditions.POISONED]
    assert state.value[Conditions.INVISIBLE]

    state = ConditionState().jax()
    state = state + ConditionState(poisoned=True).jax() + ConditionState(invisible=True).jax() + ConditionState(poisoned=False).jax()
    assert not state.value[Conditions.POISONED]
    assert state.value[Conditions.INVISIBLE]


def test_map_reduce():
    condition = map_reduce(ConditionState(poisoned=True).jax())
    assert condition.hitroll.dtype == jnp.int8
    assert condition.hitroll == RollType.DISADVANTAGE


def test_stack_reduce_poison_exhaustion():
    poisoned = ConditionState(poisoned=True).jax()
    exhaustion = ConditionState(exhaustion=True).jax()
    conditions = poisoned + exhaustion
    reduced = map_reduce(conditions)
    assert reduced.hitroll == RollType.DISADVANTAGE
    assert reduced.target_hitroll == RollType.NORMAL
    assert reduced.melee_target_hitroll == RollType.NORMAL
    assert reduced.incapacitated == False
    assert jnp.allclose(reduced.ability_check, RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw[Abilities.WIS], RollType.DISADVANTAGE)
    assert jnp.allclose(reduced.saving_throw_fail, False)
    assert reduced.melee_target_auto_crit == False
    assert jnp.allclose(reduced.damage_resistance, 1.)


def test_stack_reduce_blinded_invisible():
    blinded = ConditionState(blinded=True).jax()
    invisible = ConditionState(invisible=True).jax()
    conditions = blinded + invisible
    reduced = map_reduce(conditions)
    assert reduced.hitroll == RollType.NORMAL
    assert reduced.target_hitroll == RollType.NORMAL
    assert reduced.melee_target_hitroll == RollType.NORMAL
    assert reduced.incapacitated == False
    assert jnp.allclose(reduced.ability_check, RollType.NORMAL)
    assert jnp.allclose(reduced.saving_throw, RollType.NORMAL)
    assert jnp.allclose(reduced.saving_throw_fail, False)
    assert reduced.melee_target_auto_crit == False
    assert jnp.allclose(reduced.damage_resistance, 1.)


def test_attack_condition_modifiers():
    source = map_reduce(ConditionState(poisoned=True).jax())
    target = map_reduce(ConditionState(paralyzed=True).jax())
    hitroll_type_mod = hitroll_adv_dis(source, target)
    assert hitroll_type_mod == RollType.NORMAL
    assert target.melee_target_auto_crit

    source = map_reduce(ConditionState(invisible=True).jax())
    target = map_reduce(ConditionState(invisible=True).jax())
    hitroll_type_mod = hitroll_adv_dis(source, target)
    assert hitroll_type_mod == RollType.NORMAL

