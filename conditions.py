import chex
import dataclasses
from dataclasses import field
from typing import List
from constants import Abilities, DamageType, Conditions
import jax.numpy as jnp
from character import JaxStringArray
from dice import RollType, ad_rule
import jax


@dataclasses.dataclass
class ConditionMod:
    name: str
    condition: int
    hitroll: RollType = RollType.NORMAL
    target_hitroll: RollType = RollType.NORMAL
    incapacitated: bool = False
    ability_check: List[RollType] = field(default_factory=lambda: [RollType.NORMAL] * len(Abilities))
    saving_throw: List[RollType] = field(default_factory=lambda: [RollType.NORMAL] * len(Abilities))
    saving_throw_fail: List[bool] = field(default_factory=lambda: [False] * len(Abilities))
    melee_hitroll: RollType = RollType.NORMAL
    melee_damage_multiplier: float = 1.
    damage_resistance: List[float] = field(default_factory=lambda: [1.] * len(DamageType))


@chex.dataclass
class ConditionModArray:
    name: JaxStringArray
    condition: jnp.int32
    hitroll: jnp.int8
    target_hitroll: jnp.int8
    incapacitated: jnp.bool
    ability_check: jnp.int8
    saving_throw: jnp.int8
    saving_throw_fail: jnp.bool
    melee_hitroll: jnp.int8
    melee_damage_multiplier: jnp.float16
    damage_resistance: jnp.float16


FAIL_STR_DEX = [True, True, False, False, False, False]
RESIST_ALL_DMG = [0.5] * len(DamageType)
IMMUNE_POISON_RESIST_ALL = [0.5] * len(DamageType)
IMMUNE_POISON_RESIST_ALL[DamageType.POISON] = 0.
DISADVANTAGE_DEX=[RollType.NORMAL] * len(Abilities)
DISADVANTAGE_DEX[Abilities.DEX] = RollType.DISADVANTAGE
DISADVANTAGE_ALL = [RollType.DISADVANTAGE]* len(Abilities)


CONDITIONS = sorted([
    ConditionMod('blinded', Conditions.BLINDED, hitroll=RollType.DISADVANTAGE, target_hitroll=RollType.ADVANTAGE),
    ConditionMod('charmed', Conditions.CHARMED),
    ConditionMod('deafened', Conditions.DEAFENED),
    ConditionMod('exhausted', Conditions.EXHAUSTION, hitroll=RollType.DISADVANTAGE, saving_throw=DISADVANTAGE_ALL, ability_check=DISADVANTAGE_ALL),
    ConditionMod('frightened', Conditions.FRIGHTENED, ability_check=DISADVANTAGE_ALL, hitroll=RollType.DISADVANTAGE),
    ConditionMod('grappled', Conditions.GRAPPLED),
    ConditionMod('incapacitated', Conditions.INCAPACITATED, incapacitated=True),
    ConditionMod('invisible', Conditions.INVISIBLE, hitroll=RollType.ADVANTAGE, target_hitroll=RollType.DISADVANTAGE),
    ConditionMod('paralysed', Conditions.PARALYZED, target_hitroll=RollType.ADVANTAGE, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, melee_damage_multiplier=2.),
    ConditionMod('petrified', Conditions.PETRIFIED, incapacitated=True, target_hitroll=RollType.ADVANTAGE, saving_throw_fail=FAIL_STR_DEX, damage_resistance=IMMUNE_POISON_RESIST_ALL),
    ConditionMod('poisoned', Conditions.POISONED, hitroll=RollType.DISADVANTAGE, ability_check=DISADVANTAGE_ALL),
    ConditionMod('prone', Conditions.PRONE, hitroll=RollType.DISADVANTAGE, melee_hitroll=RollType.ADVANTAGE, target_hitroll=RollType.DISADVANTAGE),
    ConditionMod('restrained', Conditions.RESTRAINED, hitroll=RollType.DISADVANTAGE, target_hitroll=RollType.ADVANTAGE, saving_throw=DISADVANTAGE_DEX),
    ConditionMod('stunned', Conditions.STUNNED, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, target_hitroll=RollType.ADVANTAGE),
    ConditionMod('Unconscious', Conditions.UNCONSCIOUS, incapacitated=True, saving_throw_fail=FAIL_STR_DEX, target_hitroll=RollType.ADVANTAGE, melee_damage_multiplier=2.)
], key=lambda item: item.condition.value)


def reduce_condition_mod_arrays(a: ConditionModArray, axis=0) -> ConditionModArray:
    """
    Reduces two ConditionModArray instances into a single combined instance.
    For use with jax.tree.reduce to combine multiple condition modifiers.
    Handles JAX arrays appropriately for each field.
    """
    a: ConditionModArray = jax.tree.map(lambda a: jnp.moveaxis(a, axis, 0), a)
    return ConditionModArray(
        # For name, we'll keep the first one since combining strings in JAX is complex
        name=a.name,
        condition=a.condition,
        hitroll=ad_rule(a.hitroll),
        target_hitroll=ad_rule(a.target_hitroll),
        melee_hitroll=ad_rule(a.melee_hitroll),
        incapacitated=a.incapacitated.any(0),
        ability_check=ad_rule(a.ability_check),
        saving_throw=ad_rule(a.saving_throw),
        saving_throw_fail=a.saving_throw_fail.any(),
        melee_damage_multiplier=jnp.max(a.melee_damage_multiplier),
        damage_resistance=jnp.prod(a.damage_resistance, axis=0)
    )