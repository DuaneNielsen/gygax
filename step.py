import dataclasses

import jax
import jax.numpy as jnp
import pgx
import chex
from enum import IntEnum, auto
from typing import List, Dict
from pgx import EnvId
from character import CharacterExtra, stack_party
from constants import HitrollType, N_PLAYERS, N_CHARACTERS
from dnd5e import ActionTuple, decode_action
from pgx.core import Array
from character import JaxStringArray


class Ability(IntEnum):
    STR = 0
    DEX = auto()
    CON = auto()
    INT = auto()
    WIS = auto()
    CHA = auto()


class DamageType(IntEnum):
    SLASHING = 0
    FORCE = auto()


class Conditions(IntEnum):
    POISONED = 0
    PARALYZED = 1


class SaveFreq(IntEnum):
    END_TARGET_TURN = 0


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
    condition_duration: jnp.int8
    can_save: jnp.bool  # can the target use a saving throw
    save: jnp.int8  # which save the target uses
    use_save_dc: jnp.bool  # use the action DC instead of the characters
    save_dc: jnp.int8  # the item or spell dc to use if use_save_dc is set
    save_mod: jnp.float16  # damage multiplier on successful save
    save_freq: jnp.int8  # end of targets turn, on damage, none, etc
    bonus_attacks: jnp.int8  # bonus attacks can be applied to any weapon
    bonus_spell_attacks: jnp.int8  # can only use the recently cast spell
    recurring_damage: jnp.float16  # edge case where recurring damage is different to the main damage


@dataclasses.dataclass
class ActionEntry:
    name: str
    damage: float
    damage_type: int
    req_hitroll: bool
    hitroll_type: HitrollType
    ability_mod_damage: bool
    inflicts_condition: bool
    condition: Conditions
    condition_duration: int  # rounds ( 6 seconds )
    can_save: bool
    save: Ability
    use_save_dc: bool
    save_dc: int
    save_mod: float
    save_freq: SaveFreq
    bonus_attacks: int
    bonus_spell_attacks: int
    recurring_damage: float


def item(name: str,
         damage: float,
         damage_type: DamageType) -> ActionEntry:
    return ActionEntry(
        name=name,
        damage=damage,
        damage_type=damage_type,
        req_hitroll=True,
        hitroll_type=HitrollType.MELEE,
        ability_mod_damage=True,
        inflicts_condition=False,
        condition=Conditions.POISONED,  # placeholder
        condition_duration=0,
        can_save=False,
        save=Ability.CON,  # placeholder
        use_save_dc=False,
        save_dc=10,
        save_mod=1.0,
        save_freq=SaveFreq.END_TARGET_TURN,
        bonus_attacks=0,
        bonus_spell_attacks=0.,
        recurring_damage=0.
    )


def spell(name: str,
          damage: float,
          damage_type: DamageType,
          req_hitroll=False,
          hitroll_type=HitrollType.SPELL,
          ability_mod_damage=True,
          inflicts_condition=False,
          condition=Conditions.POISONED,  # placeholder
          condition_duration=0,
          can_save=False,
          save=Ability.CON,  # placeholder
          use_save_dc=False,
          save_dc=10,
          save_mod=0.5,
          save_freq=SaveFreq.END_TARGET_TURN,
          bonus_attacks=0,
          bonus_spell_attacks=0.,
          recurring_damage=0.
          ) -> ActionEntry:
    return ActionEntry(
        name,
        damage,
        damage_type,
        req_hitroll,
        hitroll_type,
        ability_mod_damage,
        inflicts_condition,
        condition,  # placeholder
        condition_duration,
        can_save,
        save,  # placeholder
        use_save_dc,
        save_dc,
        save_mod,
        save_freq,
        bonus_attacks,
        bonus_spell_attacks,
        recurring_damage
    )


action_table = [
    item('longsword', damage=4.5, damage_type=DamageType.SLASHING),
    spell('eldrich-blast', 5.5, DamageType.FORCE, True)
]

action_lookup = {entry.name: i for i, entry in enumerate(action_table)}


def load_action(action: ActionEntry):
    return ActionArray(
        name=JaxStringArray.str_to_uint8_array(action.name),
        damage=jnp.float16(action.damage),
        damage_type=jnp.int8(action.damage_type),
        req_hitroll=jnp.bool(action.req_hitroll),
        hitroll_type=jnp.int8(action.hitroll_type),
        ability_mod_damage=jnp.bool(action.ability_mod_damage),
        inflicts_condition=jnp.bool(action.inflicts_condition),
        condition=jnp.int8(action.condition),
        condition_duration=jnp.int8(action.condition_duration),
        can_save=jnp.bool(action.can_save),
        save=jnp.int8(action.save),
        use_save_dc=jnp.bool(action.use_save_dc),
        save_dc=jnp.int8(action.save_dc),
        save_mod=jnp.float16(action.save_mod),
        save_freq=jnp.int8(action.save_freq),
        bonus_attacks=jnp.int8(action.bonus_attacks),
        bonus_spell_attacks=jnp.int8(action.bonus_spell_attacks),
        recurring_damage=jnp.float16(action.recurring_damage),
    )


action_table = [load_action(a) for a in action_table]
action_table = jax.tree.map(lambda *x: jnp.stack(x), *action_table)


@chex.dataclass
class Character:
    name: JaxStringArray
    hp: jnp.float16
    ac: jnp.int8
    prof_bonus: jnp.int8
    ability_mods: jnp.int8
    attack_ability_mods: jnp.int8
    save_bonus: jnp.int8
    damage_type_mul: jnp.float16


import pgx._src.struct


@pgx._src.struct.dataclass
class State(pgx.State):
    character: Character
    pos: jnp.uint8
    current_player: jnp.int8
    observation: jnp.array
    rewards: jnp.float32
    terminated: jnp.bool
    truncated: jnp.bool
    legal_action_mask: jnp.bool
    _step_count: jnp.int32

    @property
    def env_id(self) -> EnvId:
        return "fast_poc"


# stub init to help for testing, move to test harness later
def init(party: Dict[str, Dict[str, CharacterExtra]]):
    characters = stack_party(party, Character)
    return State(
        character=characters,
        pos=jnp.tile(jnp.arange(N_CHARACTERS, dtype=jnp.uint8), (N_PLAYERS, 1)),
        current_player=jnp.int32(0),
        observation=jnp.array([0]),
        rewards=jnp.float32(0),
        terminated=jnp.bool(False),
        truncated=jnp.bool(False),
        legal_action_mask=jnp.array([True, False]),
        _step_count=jnp.int32(0),
    )


# so much for damage, think about conditions and conditional probability of successive saves next
# advantage on saves and attack rolls
debug = True


def step(state: State, action: Array):
    action = decode_action(action, state.current_player, state.pos)
    source = jax.tree.map(lambda x: x[*action.source], state.character)
    weaponspell = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target = jax.tree.map(lambda x: x[*action.target], state.character)

    hitroll = 20 - target.ac + source.prof_bonus + source.attack_ability_mods[weaponspell.hitroll_type]
    hitroll = jnp.float16(hitroll).clip(1, 20) / 20
    hitroll_mul = jnp.where(weaponspell.req_hitroll, hitroll, jnp.ones_like(hitroll))

    save_dc = source.prof_bonus + source.attack_ability_mods[weaponspell.hitroll_type]
    save_dc = jnp.where(weaponspell.use_save_dc, weaponspell.save_dc, save_dc)
    target_bonus_save = target.prof_bonus * target.save_bonus[weaponspell.save] + target.ability_mods[weaponspell.save]
    save = 8 + save_dc - target_bonus_save
    save = jnp.float16(save).clip(1, 20) / 20
    save_mul = jnp.where(weaponspell.can_save, save, jnp.ones_like(save))

    damage = weaponspell.damage * hitroll_mul * save_mul * target.damage_type_mul[weaponspell.damage_type]
    state.character.hp = state.character.hp.at[*action.target].set(state.character.hp[*action.target] - damage)

    if debug:
        source_name = JaxStringArray.uint8_array_to_str(source.name)
        target_name = JaxStringArray.uint8_array_to_str(target.name)
        action_name = JaxStringArray.uint8_array_to_str(weaponspell.name)
        print(f'{source_name}, {action_name}, {target_name}')

    return state