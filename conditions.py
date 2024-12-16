import chex
import dataclasses
from dataclasses import field
from typing import List
from constants import Abilities, DamageType, Conditions
import jax.numpy as jnp
from character import convert, JaxStringArray


@dataclasses.dataclass
class ConditionMod:
    name: str
    condition: int
    hitroll_advantage: bool = False
    hitroll_disadvantage : bool = False
    target_hitroll_advantage: bool = False
    target_hitroll_disadvantage: bool = False
    incapacitated: bool = False
    ability_check_disadvantage: List[bool] = field(default_factory=lambda: [False] * len(Abilities))
    saving_throw_disadvantage: List[bool] = field(default_factory=lambda: [False] * len(Abilities))
    saving_throw_fail: List[bool] = field(default_factory=lambda: [False] * len(Abilities))
    melee_hitroll_advantage: bool = False
    melee_damage_multiplier: float = 1.
    damage_resistance: List[float] = field(default_factory=lambda: [False] * len(DamageType))


@chex.dataclass
class ConditionModArray:
    name: JaxStringArray
    condition: jnp.int32
    hitroll_advantage: jnp.bool
    hitroll_disadvantage : jnp.bool
    target_hitroll_advantage: jnp.bool  # attacks targeting have advantage
    target_hitroll_disadvantage: jnp.bool  # attacks targeting have disadvantage
    incapacitated: jnp.bool  # cannot take actions
    ability_check_disadvantage: jnp.bool
    saving_throw_disadvantage: jnp.bool
    saving_throw_fail: jnp.bool
    melee_hitroll_advantage: jnp.bool
    melee_damage_multiplier: jnp.float16
    damage_resistance: jnp.float16


FAIL_DEX_STR = [True, True, False, False, False, False]



CONDITIONS = [
    ConditionMod('blinded', Conditions.BLINDED, hitroll_disadvantage=True, target_hitroll_advantage=True),
    ConditionMod('charmed', Conditions.CHARMED),
    ConditionMod('deafened', Conditions.DEAFENED),
    ConditionMod('frightened', Conditions.FRIGHTENED, ability_check_disadvantage=[True] * len(Abilities), hitroll_disadvantage=True),
    ConditionMod('paralysed', Conditions.PARALYZED, target_hitroll_advantage=True, incapacitated=True, saving_throw_fail=FAIL_DEX_STR, melee_damage_multiplier=2.)
]

