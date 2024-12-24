import dataclasses
from dataclasses import field
from enum import StrEnum, IntEnum, auto
from typing import List

import chex
from dnd_character.SRD import SRD
from dnd_character.equipment import _Item
from jax import numpy as jnp

from conditions import Conditions
from constants import DamageType, HitrollType, Abilities, SaveFreq
from to_jax import JaxStringArray, convert
import jax


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


@dataclasses.dataclass
class ActionEntry:
    name: str = ''
    active: bool = False
    damage: float = 0.
    damage_type: int = DamageType.FORCE
    req_hitroll: bool = False
    hitroll_type: HitrollType = HitrollType.SPELL
    hitroll_bonus: int = 0
    ability_mod_damage: bool = False
    inflicts_condition: bool = False
    condition: Conditions = Conditions.POISONED
    duration: int = 0  # rounds ( 6 seconds )
    ends_on_attack: bool = False
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
    legal_source_slots: List[bool] = field(default_factory=lambda: [True, True, True, True])
    legal_target_slots: List[List[bool]] = field(default_factory=lambda: [[False, False, False, False], [True, True, True, True]])
    aoe: bool = False
    aoe_target_slots: List[List[bool]] = field(default_factory=lambda: [[False, False, False, False], [True, True, True, True]])

    def replace(self, **kwargs) -> 'ActionEntry':
        return dataclasses.replace(self, **kwargs)

    def jax(self):
        return convert(self, ActionArray)


@chex.dataclass
class ActionArray:
    name: JaxStringArray
    active: jnp.bool
    damage: jnp.float16
    damage_type: jnp.int8
    req_hitroll: jnp.bool  #
    hitroll_type: jnp.int8  # hitroll type, melee, finesse, ranged, spell
    hitroll_bonus: jnp.int8
    ability_mod_damage: jnp.bool  # if true, add ability mod to damage
    inflicts_condition: jnp.bool
    condition: jnp.int8
    duration: jnp.int8
    ends_on_attack: jnp.bool
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
    legal_source_slots: jnp.bool
    legal_target_slots: jnp.bool
    aoe: jnp.bool
    aoe_target_slots: jnp.bool

    def __add__(self, other):
        return ActionArray(
            name=self.name,
            active=self.active,
            damage=self.damage + other.damage,
            damage_type=self.damage_type,
            req_hitroll=self.req_hitroll,
            hitroll_type=self.hitroll_type,
            hitroll_bonus=self.hitroll_bonus + other.hitroll_bonus,
            ability_mod_damage=self.ability_mod_damage,
            inflicts_condition=self.inflicts_condition,
            condition=self.condition,
            duration=self.duration + other.duration,
            ends_on_attack=self.ends_on_attack,
            can_save=self.can_save,
            save=self.save,
            use_save_dc=self.use_save_dc,
            save_dc=self.save_dc + other.save_dc,
            save_mod=self.save_mod,
            save_freq=self.save_freq,
            cum_save=self.cum_save,
            bonus_attacks=self.bonus_attacks + other.bonus_attacks,
            bonus_spell_attacks=self.bonus_spell_attacks + other.bonus_spell_attacks,
            recurring_damage=self.recurring_damage + other.recurring_damage,
            recurring_damage_save_mod=self.recurring_damage_save_mod,
            recurring_damage_hitroll=self.recurring_damage_hitroll,
            req_concentration=self.req_concentration
        )

MELEE_SLOTS = jnp.bool([False, False, True, True])
RANGED_SLOTS = jnp.bool([True, True, False, False])

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
    ends_on_attack=False,
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


melee_weapon = item.replace(
    legal_source_slots=MELEE_SLOTS,
)
ranged_weapon = item.replace(
    legal_source_slots=RANGED_SLOTS,
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
    ends_on_attack=False,
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
melee_spell = spell.replace(
    legal_source_slots=MELEE_SLOTS
)

AOE_ALL_ENEMY = jnp.bool([
    [False, False, False, False],
    [True, True, True, True]
])

AOE_MELEE_ENEMY = jnp.bool([
    [False, False, False, False],
    [False, False, True, True]
])

AOE_ALL_FRIENDLY = jnp.bool([
    [False, False, False, False],
    [True, True, True, True]
])


action_table = [
    spell.replace(name='end-turn'),
    melee_weapon.replace(name='longsword', damage=4.5, damage_type=DamageType.SLASHING),
    melee_weapon.replace(name='longsword-two-hand', damage=5.5, damage_type=DamageType.SLASHING),
    melee_weapon.replace(name='rapier', damage=4.5, damage_type=DamageType.PIERCING),
    melee_weapon.replace(name='rapier-finesse', hitroll_type=HitrollType.FINESSE, damage=4.5, damage_type=DamageType.PIERCING),
    ranged_weapon.replace(name='longbow', hitroll_type=HitrollType.RANGED, damage=4.5, damage_type=DamageType.PIERCING),
    ranged_weapon.replace(name='shortbow', hitroll_type=HitrollType.RANGED, damage=3.5, damage_type=DamageType.PIERCING),
    spell.replace(name='eldrich-blast', damage=5.5, damage_type=DamageType.FORCE, req_hitroll=True),
    spell.replace(name='agonizing-blast', damage=5.5, damage_type=DamageType.FORCE, req_hitroll=True,
                  ability_mod_damage=True),
    melee_spell.replace(name='poison-spray', damage=6.5, damage_type=DamageType.POISON, can_save=True, save=Abilities.CON,
                  save_mod=0., aoe=True, aoe_target_slots=AOE_MELEE_ENEMY),
    melee_spell.replace(name='burning-hands', damage=3 * 3.5, damage_type=DamageType.FIRE, can_save=True, save=Abilities.DEX,
                  save_mod=0.5, aoe=True, aoe_target_slots=AOE_MELEE_ENEMY),
    spell.replace(name='acid-arrow', damage=4 * 2.5, damage_type=DamageType.ACID, req_hitroll=True, duration=1,
                  recurring_damage=2 * 2.5, recurring_damage_save_mod=0),
    spell.replace(name='hold-person', inflicts_condition=True, condition=Conditions.PARALYZED, can_save=True,
                  save=Abilities.WIS, duration=10, req_concentration=True),
    spell.replace(name='guided-bolt', damage=4 * 3.5, damage_type=DamageType.RADIANT, inflicts_condition=True,
                  condition=Conditions.ILLUMINATED, ends_on_attack=True),
    spell.replace(name='fireball', damage=8 * 3.5, damage_type=DamageType.FIRE, can_save=True, save=Abilities.DEX, aoe=True, aoe_target_slots=AOE_ALL_ENEMY)
]
action_table = sorted(action_table, key=lambda k: k.name)

ActionsEnum = IntEnum('ActionsEnum',{entry.name: i for i, entry in enumerate(action_table)})
action_table = [a.jax() for a in action_table]
action_table = jax.tree.map(lambda *x: jnp.stack(x), *action_table)


MAGIC_BONUS = [
    item.replace(name='+1', damage=1., hitroll_bonus=1),
    item.replace(name='+2', damage=2., hitroll_bonus=2),
    item.replace(name='+3', damage=3., hitroll_bonus=3),
]

# Actions = {entry.name: i for i, entry in enumerate(action_table)}
MAGIC_BONUS = sorted(MAGIC_BONUS, key=lambda k: k.name)
MagicBonus = IntEnum('ActionsEnum', {entry.name: i for i, entry in enumerate(MAGIC_BONUS)})
MAGIC_BONUS = [a.jax() for a in MAGIC_BONUS]
MAGIC_BONUS = jax.tree.map(lambda *x: jnp.stack(x), *MAGIC_BONUS)


def lookup_magic_bonus(bonus: str) -> ActionArray:
    i = MagicBonus[bonus]
    return jax.tree.map(lambda x: x[i], MAGIC_BONUS)