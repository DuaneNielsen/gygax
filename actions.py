import dataclasses
from enum import StrEnum, IntEnum

import chex
from dnd_character.SRD import SRD
from dnd_character.equipment import _Item
from jax import numpy as jnp

from conditions import Conditions
from constants import DamageType, HitrollType, Abilities, SaveFreq
from to_jax import JaxStringArray, convert
import jax


@dataclasses.dataclass
class ActionEntry:
    name: str = ''
    damage: float = 0.
    damage_type: int = DamageType.FORCE
    req_hitroll: bool = False
    hitroll_type: HitrollType = HitrollType.SPELL
    ability_mod_damage: bool = False
    inflicts_condition: bool = False
    condition: Conditions = Conditions.POISONED
    duration: int = 0  # rounds ( 6 seconds )
    can_save: bool = False
    save: Abilities = Abilities.CON
    use_save_dc: bool = False
    save_dc: int = 0
    save_mod: float = 0.
    cum_save: float = 0.
    save_freq: SaveFreq = SaveFreq.END_TARGET_TURN
    bonus_attacks: int = 0
    bonus_spell_attacks: int = 0
    recurring_damage: float = 0.
    recurring_damage_save_mod: float = 0.
    recurring_damage_hitroll: float = 1.
    req_concentration: bool = False

    def replace(self, **kwargs) -> 'ActionEntry':
        return dataclasses.replace(self, **kwargs)


class WeaponRange(StrEnum):
    MELEE = 'Melee'
    RANGED = 'Ranged'


unarmed = _Item(
    index="unarmed",
    name="unarmed",
    equipment_category=SRD('/api/equipment-categories/weapon'),
    weapon_category='Simple',
    weapon_range=WeaponRange.MELEE,
    range={'normal': 5},
    damage={
        'damage_dice': '1d1',
        'damage_type': SRD('/api/damage-types/bludgeoning')
    },
    contents=None,
    cost=0,
    desc='unarmed attack',
    properties=[],
    special=None,
    url=None
)


@chex.dataclass
class ActionArray:
    name: JaxStringArray
    damage: jnp.float16
    damage_type: jnp.int8
    req_hitroll: jnp.bool  #
    hitroll_type: jnp.int8  # hitroll type, melee, finesse, ranged, spell
    ability_mod_damage: jnp.bool  # if true, add ability mod to damage
    inflicts_condition: jnp.bool
    condition: jnp.int8
    duration: jnp.int8
    can_save: jnp.bool  # can the target use a saving throw
    save: jnp.int8  # which save the target uses
    use_save_dc: jnp.bool  # use the action DC instead of the characters
    save_dc: jnp.int8  # the item or spell dc to use if use_save_dc is set
    save_mod: jnp.float16  # damage multiplier on successful save
    save_freq: jnp.int8  # end of targets turn, on damage, none, etc
    cum_save: jnp.float16  # to track the effect of cumulative saves over time
    bonus_attacks: jnp.int8  # bonus attacks can be applied to any weapon
    bonus_spell_attacks: jnp.int8  # can only use the recently cast spell
    recurring_damage: jnp.float16  # edge case where recurring damage is different to the main damage
    recurring_damage_save_mod: jnp.float16  # recurring damage reduction on save
    recurring_damage_hitroll: jnp.float16  # store the hitroll to compute concentration checks
    req_concentration: jnp.bool


item = ActionEntry(
    'item',
    damage=0.,
    damage_type=DamageType.SLASHING,
    req_hitroll=True,
    hitroll_type=HitrollType.MELEE,
    ability_mod_damage=True,
    inflicts_condition=False,
    condition=Conditions.POISONED,  # placeholder
    duration=0,
    can_save=False,
    save=Abilities.CON,  # placeholder
    use_save_dc=False,
    save_dc=10,
    save_mod=0.5,
    save_freq=SaveFreq.END_TARGET_TURN,
    cum_save=0.,
    bonus_attacks=0,
    bonus_spell_attacks=0.,
    recurring_damage=0.,
    req_concentration=False,
)
spell = ActionEntry(
    'spell',
    damage=0.,
    damage_type=DamageType.FORCE,
    req_hitroll=False,
    hitroll_type=HitrollType.SPELL,
    ability_mod_damage=False,
    inflicts_condition=False,
    condition=Conditions.POISONED,  # placeholder
    duration=0,
    can_save=False,
    save=Abilities.CON,  # placeholder
    use_save_dc=False,
    save_dc=10,
    save_mod=0.5,
    cum_save=0.,
    save_freq=SaveFreq.END_TARGET_TURN,
    bonus_attacks=0,
    bonus_spell_attacks=0.,
    recurring_damage=0.,
    recurring_damage_save_mod=0.,
    req_concentration=False
)
action_table = [
    spell.replace(name='end-turn'),
    item.replace(name='longsword', damage=4.5, damage_type=DamageType.SLASHING),
    item.replace(name='longsword-two-hand', damage=5.5, damage_type=DamageType.SLASHING),
    item.replace(name='rapier', damage=4.5, damage_type=DamageType.PIERCING),
    item.replace(name='rapier-finesse', hitroll_type=HitrollType.FINESSE, damage=4.5, damage_type=DamageType.PIERCING),
    item.replace(name='longbow', hitroll_type=HitrollType.RANGED, damage=4.5, damage_type=DamageType.PIERCING),
    item.replace(name='shortbow', hitroll_type=HitrollType.RANGED, damage=3.5, damage_type=DamageType.PIERCING),
    spell.replace(name='eldrich-blast', damage=5.5, damage_type=DamageType.FORCE, req_hitroll=True),
    spell.replace(name='agonizing-blast', damage=5.5, damage_type=DamageType.FORCE, req_hitroll=True,
                  ability_mod_damage=True),
    spell.replace(name='poison-spray', damage=6.5, damage_type=DamageType.POISON, can_save=True, save=Abilities.CON,
                  save_mod=0.),
    spell.replace(name='burning-hands', damage=3 * 3.5, damage_type=DamageType.FIRE, can_save=True, save=Abilities.DEX,
                  save_mod=0.5),
    spell.replace(name='acid-arrow', damage=4 * 2.5, damage_type=DamageType.ACID, req_hitroll=True, duration=1,
                  recurring_damage=2 * 2.5, recurring_damage_save_mod=0),
    spell.replace(name='hold-person', inflicts_condition=True, condition=Conditions.PARALYZED, can_save=True,
                  save=Abilities.WIS, duration=10, req_concentration=True)
]

Actions = {entry.name: i for i, entry in enumerate(action_table)}
ActionsEnum = IntEnum('ActionsEnum', [a.name for a in action_table])
action_table = [convert(a, ActionArray) for a in action_table]
action_table = jax.tree.map(lambda *x: jnp.stack(x), *action_table)