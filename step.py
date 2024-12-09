import dataclasses
import jax
import jax.numpy as jnp
import pgx
import chex
from enum import IntEnum, auto
from typing import List, Dict
from pgx import EnvId

import constants
from character import CharacterExtra, stack_party, ActionEntry
from constants import HitrollType, N_PLAYERS, N_CHARACTERS, Conditions, Abilities, SaveFreq
from dnd5e import ActionTuple, decode_action
from pgx.core import Array
from character import JaxStringArray, DamageType
import numpy as np


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
    cum_save: jnp.float16  # to track the effect of cumulative saves over time
    bonus_attacks: jnp.int8  # bonus attacks can be applied to any weapon
    bonus_spell_attacks: jnp.int8  # can only use the recently cast spell
    recurring_damage: jnp.float16  # edge case where recurring damage is different to the main damage


def item(name: str,
         damage: float,
         damage_type: DamageType,
         req_hitroll=True,
         hitroll_type=HitrollType.MELEE,
         ability_mod_damage=True,
         inflicts_condition=False,
         condition=Conditions.POISONED,  # placeholder
         condition_duration=0,
         can_save=False,
         save=Abilities.CON,  # placeholder
         use_save_dc=False,
         save_dc=10,
         save_mod=0.5,
         save_freq=SaveFreq.END_TARGET_TURN,
         cum_save=0.,
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
        cum_save,
        bonus_attacks,
        bonus_spell_attacks,
        recurring_damage
    )


def spell(name: str,
          damage=0.,
          damage_type=DamageType.FORCE,
          req_hitroll=False,
          hitroll_type=HitrollType.SPELL,
          ability_mod_damage=False,
          inflicts_condition=False,
          condition=Conditions.POISONED,  # placeholder
          condition_duration=0,
          can_save=False,
          save=Abilities.CON,  # placeholder
          use_save_dc=False,
          save_dc=10,
          save_mod=0.5,
          cum_save=0.,
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
        cum_save,
        bonus_attacks,
        bonus_spell_attacks,
        recurring_damage
    )


action_table = [
    spell('end_turn'),
    item('longsword', damage=4.5, damage_type=DamageType.SLASHING),
    item('longbow', hitroll_type=HitrollType.RANGED, damage=4.5, damage_type=DamageType.PIERCING),
    spell('eldrich-blast', 5.5, DamageType.FORCE, True),
    spell('poison-spray', 6.5, DamageType.POISON, can_save=True, save=Abilities.CON, save_mod=0.),
    spell('burning-hands', 3.5 * 3, DamageType.FIRE, can_save=True, save=Abilities.DEX, save_mod=0.5),
    spell('hold-person', inflicts_condition=True, condition=Conditions.PARALYZED, can_save=True, save=Abilities.WIS)
]

Actions = {entry.name: i for i, entry in enumerate(action_table)}
ActionsEnum = IntEnum('ActionsEnum', [a.name for a in action_table])


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
        cum_save=jnp.float32(action.cum_save),
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
    conditions: jnp.bool
    effects: ActionArray


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


def step_to_str(source: Character, target: Character, weaponspell: ActionArray, damage: jnp.float16,
                save_fail: jnp.float16):
    source_name = JaxStringArray.uint8_array_to_str(source.name)
    target_name = JaxStringArray.uint8_array_to_str(target.name)
    action_name = JaxStringArray.uint8_array_to_str(weaponspell.name)
    save_fail = np.array(save_fail)
    damage = np.array(damage)
    hitroll_type = weaponspell.hitroll_type.item()

    save_msg = f' saved {save_fail:.2f}' if weaponspell.can_save else ''

    if hitroll_type in {HitrollType.MELEE, HitrollType.FINESSE}:
        return f'{source_name} hit {target_name} with {action_name} for {damage:.2f}' + save_msg
    elif hitroll_type == HitrollType.SPELL:
        return f'{source_name} cast {action_name} on {target_name} for {damage:.2f}' + save_msg
    elif hitroll_type == HitrollType.RANGED:
        return f'{source_name} shot {target_name} with {action_name} for {damage:.2f}' + save_msg


def print_step(*args):
    print(step_to_str(*args))


def step(state: State, action: Array):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    source: Character = jax.tree.map(lambda x: x[*action.source], state.character)
    weapon: ActionArray = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[*action.target], state.character)

    source_ability_bonus = source.attack_ability_mods[weapon.hitroll_type]

    hitroll = 20 - target.ac + source.prof_bonus + source_ability_bonus
    hitroll = jnp.float16(hitroll).clip(1, 20) / 20
    hitroll_mul = jnp.where(weapon.req_hitroll, hitroll, jnp.ones_like(hitroll))

    save_dc = 8 + source.prof_bonus + source_ability_bonus
    save_dc = jnp.where(weapon.use_save_dc, weapon.save_dc, save_dc)
    target_bonus_save = target.prof_bonus * target.save_bonus[weapon.save] + target.ability_mods[weapon.save]
    save_hurdle = save_dc - target_bonus_save
    save_fail_prob = jnp.float16(save_hurdle).clip(1, 20) / 20
    save_mul = save_fail_prob + (1 - save_fail_prob) * weapon.save_mod
    save_mul = jnp.where(weapon.can_save, save_mul, jnp.ones_like(save_mul))

    damage = weapon.damage + jnp.where(weapon.ability_mod_damage, source_ability_bonus, 0)
    damage = damage * hitroll_mul * save_mul * target.damage_type_mul[weapon.damage_type]
    state.character.hp = state.character.hp.at[*action.target].set(state.character.hp[*action.target] - damage)

    # condition < 0.5 means the condition stays in effect
    save_fail_prob = jnp.where(weapon.inflicts_condition, save_fail_prob, 0.)
    state.character.conditions = state.character.conditions.at[*action.target, weapon.condition].set(save_fail_prob)

    # perform end of turn actions
    source_conditions = source.conditions

    if debug:
        jax.debug.callback(print_step, source, target, weapon, damage, save_fail_prob)

    return state
