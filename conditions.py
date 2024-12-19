from enum import IntEnum, auto

import chex
import dataclasses
from dataclasses import field
from typing import List
from constants import Abilities, DamageType
import jax.numpy as jnp
from to_jax import convert, JaxStringArray
from dice import RollType, ad_rule
import jax


@dataclasses.dataclass
class ConditionMod:
    name: str
    condition: int
    hitroll: RollType = RollType.NORMAL
    target_hitroll: RollType = RollType.NORMAL
    melee_target_hitroll: RollType = RollType.NORMAL
    melee_target_auto_crit: bool = False
    incapacitated: bool = False
    ability_check: List[RollType] = field(default_factory=lambda: [RollType.NORMAL] * len(Abilities))
    saving_throw: List[RollType] = field(default_factory=lambda: [RollType.NORMAL] * len(Abilities))
    saving_throw_damage_type: List[RollType] = field(default_factory=lambda: [RollType.NORMAL] * len(DamageType))
    saving_throw_fail: List[bool] = field(default_factory=lambda: [False] * len(Abilities))
    damage_resistance: List[float] = field(default_factory=lambda: [1.] * len(DamageType))


@chex.dataclass
class ConditionModArray:
    name: JaxStringArray
    condition: jnp.int32
    hitroll: jnp.int8
    target_hitroll: jnp.int8
    melee_target_hitroll: jnp.int8
    melee_target_auto_crit: jnp.bool
    incapacitated: jnp.bool
    ability_check: jnp.int8
    saving_throw: jnp.int8
    saving_throw_fail: jnp.bool
    damage_resistance: jnp.float16


FAIL_STR_DEX = [True, True, False, False, False, False]
RESIST_ALL_DMG = [0.5] * len(DamageType)
RESIST_POISON = [1.] * len(DamageType)
RESIST_POISON[DamageType.POISON] = [0.5]
IMMUNE_POISON_RESIST_ALL = [0.5] * len(DamageType)
IMMUNE_POISON_RESIST_ALL[DamageType.POISON] = 0.
DISADVANTAGE_DEX=[RollType.NORMAL] * len(Abilities)
DISADVANTAGE_DEX[Abilities.DEX] = RollType.DISADVANTAGE
DISADVANTAGE_ALL = [RollType.DISADVANTAGE]* len(Abilities)


class Conditions(IntEnum):
    BLINDED = 0
    CHARMED = auto()
    DEAFENED = auto()
    EXHAUSTION = auto()
    FRIGHTENED = auto()
    GRAPPLED = auto()
    INCAPACITATED = auto()
    INVISIBLE = auto()
    PARALYZED = auto()
    PETRIFIED = auto()
    POISONED = auto()
    PRONE = auto()
    RESTRAINED = auto()
    STUNNED = auto()
    UNCONSCIOUS = auto()
    PROTECTED_POISON = auto()


CONDITIONS = sorted([
    ConditionMod('blinded', Conditions.BLINDED, hitroll=RollType.DISADVANTAGE, target_hitroll=RollType.ADVANTAGE),
    ConditionMod('charmed', Conditions.CHARMED),
    ConditionMod('deafened', Conditions.DEAFENED),
    ConditionMod('exhausted', Conditions.EXHAUSTION, hitroll=RollType.DISADVANTAGE, saving_throw=DISADVANTAGE_ALL, ability_check=DISADVANTAGE_ALL),
    ConditionMod('frightened', Conditions.FRIGHTENED, ability_check=DISADVANTAGE_ALL, hitroll=RollType.DISADVANTAGE),
    ConditionMod('grappled', Conditions.GRAPPLED),
    ConditionMod('incapacitated', Conditions.INCAPACITATED, incapacitated=True),
    ConditionMod('invisible', Conditions.INVISIBLE, hitroll=RollType.ADVANTAGE, target_hitroll=RollType.DISADVANTAGE),
    ConditionMod('paralysed', Conditions.PARALYZED, target_hitroll=RollType.ADVANTAGE, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, melee_target_auto_crit=True),
    ConditionMod('petrified', Conditions.PETRIFIED, incapacitated=True, target_hitroll=RollType.ADVANTAGE, saving_throw_fail=FAIL_STR_DEX, damage_resistance=IMMUNE_POISON_RESIST_ALL),
    ConditionMod('poisoned', Conditions.POISONED, hitroll=RollType.DISADVANTAGE, ability_check=DISADVANTAGE_ALL),
    ConditionMod('prone', Conditions.PRONE, hitroll=RollType.DISADVANTAGE, melee_target_hitroll=RollType.ADVANTAGE, target_hitroll=RollType.DISADVANTAGE),
    ConditionMod('restrained', Conditions.RESTRAINED, hitroll=RollType.DISADVANTAGE, target_hitroll=RollType.ADVANTAGE, saving_throw=DISADVANTAGE_DEX),
    ConditionMod('stunned', Conditions.STUNNED, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, target_hitroll=RollType.ADVANTAGE),
    ConditionMod('unconscious', Conditions.UNCONSCIOUS, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, target_hitroll=RollType.ADVANTAGE, melee_target_auto_crit=True),
    ConditionMod('protected-poison', Conditions.PROTECTED_POISON, )
], key=lambda item: item.condition.value)
CONDITIONS = jax.tree.map(lambda *x: jnp.stack(x), *[convert(c, ConditionModArray) for c in CONDITIONS])

default_element = convert(ConditionMod(name='default', condition=-1), ConditionModArray)


def copy(x):
    reps = tuple([len(Conditions)] + [1] * len(x.shape))
    return jnp.tile(x, reps=reps)


DEFAULT = jax.tree.map(copy, default_element)


def reduce_damage_resist_vun(damage_resists, axis=0):
    """
    Damage Resistance and Vulnerability:
        Resistance and then vulnerability are applied after all other modifiers to damage.

    Resistance and Vulnerability: Here’s the order that you apply modifiers to damage:
        (1) any relevant damage immunity,
        (2) any addition or subtraction to the damage,
        (3) one relevant damage resistance
        (4) one relevant damage vulnerability.
    Returns:

    """
    return jnp.max(damage_resists, axis=axis) * jnp.min(damage_resists, axis=axis)


def reduce(a: ConditionModArray, axis=0) -> ConditionModArray:
    """
    Reduces ConditionModArray along an axis
    """
    a: ConditionModArray = jax.tree.map(lambda a: jnp.moveaxis(a, axis, 0), a)
    return ConditionModArray(
        # For name, we'll keep the first one since combining strings in JAX is complex
        name=a.name,
        condition=a.condition,
        hitroll=ad_rule(a.hitroll),
        target_hitroll=ad_rule(a.target_hitroll),
        melee_target_hitroll=ad_rule(a.melee_target_hitroll),
        melee_target_auto_crit=a.melee_target_auto_crit.any(0),
        incapacitated=a.incapacitated.any(0),
        ability_check=ad_rule(a.ability_check),
        saving_throw=ad_rule(a.saving_throw),
        saving_throw_fail=a.saving_throw_fail.any(0),
        damage_resistance=reduce_damage_resist_vun(a.damage_resistance)
    )


def map_reduce(conditions: jnp.bool) -> ConditionModArray:

    # Create mask by broadcasting conditions to match the shape of each field
    def select_values(cond_val, default_val):
        # Ensure the selection mask matches the shape of the values
        mask_shape = [len(Conditions)] + [1] * (len(cond_val.shape) - 1)
        mask = jnp.broadcast_to(conditions.reshape(mask_shape), cond_val.shape)
        return jnp.where(mask, cond_val, default_val)

    map = jax.tree.map(select_values, CONDITIONS, DEFAULT)
    return reduce(map)


def hitroll_adv_dis(source_conditions: ConditionModArray, target_conditions: ConditionModArray):
    return ad_rule(jnp.stack([source_conditions.hitroll, target_conditions.target_hitroll]))