import dataclasses
import jax
import jax.numpy as jnp
import pgx
import chex
from enum import IntEnum, auto
from typing import List, Dict
from pgx import EnvId

import character
import constants
from character import CharacterExtra, stack_party, ActionEntry
from constants import HitrollType, N_PLAYERS, N_CHARACTERS, Conditions, Abilities, SaveFreq
from dnd5e import ActionTuple, decode_action
from pgx.core import Array
from character import JaxStringArray, DamageType
import numpy as np
from functools import partial


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
    recurring_damage=0.
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
    recurring_damage_save_mod=0.
)

action_table = [
    spell.replace(name='end-turn'),
    item.replace(name='longsword', damage=4.5, damage_type=DamageType.SLASHING),
    item.replace(name='longbow', hitroll_type=HitrollType.RANGED, damage=4.5, damage_type=DamageType.PIERCING),
    spell.replace(name='eldrich-blast', damage=5.5, damage_type=DamageType.FORCE, req_hitroll=True),
    spell.replace(name='poison-spray', damage=6.5, damage_type=DamageType.POISON, can_save=True, save=Abilities.CON,
                  save_mod=0.),
    spell.replace(name='burning-hands', damage=3 * 3.5, damage_type=DamageType.FIRE, can_save=True, save=Abilities.DEX,
                  save_mod=0.5),
    spell.replace(name='acid-arrow', damage=4 * 2.5, damage_type=DamageType.ACID, req_hitroll=True, duration=1,
                  recurring_damage=2 * 2.5, recurring_damage_save_mod=0),
    spell.replace(name='hold-person', inflicts_condition=True, condition=Conditions.PARALYZED, can_save=True,
                  save=Abilities.WIS, duration=10)
]

Actions = {entry.name: i for i, entry in enumerate(action_table)}
ActionsEnum = IntEnum('ActionsEnum', [a.name for a in action_table])

action_table = [character.convert(a, ActionArray) for a in action_table]
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
    effect_active: jnp.bool
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


def step_to_str(source: Character, target: Character, weapon: ActionArray, damage: jnp.float16,
                save_fail: jnp.float16):
    source_name = JaxStringArray.uint8_array_to_str(source.name)
    target_name = JaxStringArray.uint8_array_to_str(target.name)
    action_name = JaxStringArray.uint8_array_to_str(weapon.name)
    save_fail = np.array(save_fail)
    damage = np.array(damage)
    hitroll_type = weapon.hitroll_type.item()

    save_msg = f' saved {save_fail:.2f}' if weapon.can_save else ''

    if hitroll_type in {HitrollType.MELEE, HitrollType.FINESSE}:
        return f'{source_name} hit {target_name} with {action_name} for {damage:.2f}' + save_msg
    elif hitroll_type == HitrollType.SPELL:
        return f'{source_name} cast {action_name} on {target_name} for {damage:.2f}' + save_msg
    elif hitroll_type == HitrollType.RANGED:
        return f'{source_name} shot {target_name} with {action_name} for {damage:.2f}' + save_msg


def print_step(*args):
    print(step_to_str(*args))


def save(save, save_dc, target):
    target_bonus_save = target.prof_bonus * target.save_bonus[save] + target.ability_mods[save]
    save_hurdle = save_dc - target_bonus_save
    return jnp.float16(save_hurdle).clip(1, 20) / 20


vmap_save = jax.vmap(save, in_axes=(0, 0, None))

# @chex.chexify
# @jax.jit
# def check_overflow(effect_index, target):
#     #  assert effect_index < target.effect_active.shape[-1]  # check for overflow
#     chex.assert_equal(effect_index < target.effect_active.shape[-1], True)
#     return effect_index


def update_character_if(character, cond, leaf, value):
    updated = leaf.at[*character].set(value)
    return jnp.where(cond, updated, leaf)


def step_effect(effect: ActionArray, character):
    # saving throws to resist active effects
    effect_save_fail_prob = save(effect.save, effect.save_dc, character)
    effect.cum_save = effect.cum_save * effect_save_fail_prob
    effect.cum_save = jnp.where(effect.cum_save > 0.5, effect.cum_save, jnp.zeros_like(effect.cum_save))
    effect.inflicts_condition = jnp.where(effect.inflicts_condition, effect.cum_save > 0.5, False)

    # finally decrease the duration by 1 round
    effect.duration = effect.duration - 1
    effect_active = effect.duration > 0
    effect_active = jnp.where(effect.can_save, effect.cum_save > 0.5, effect_active)  # clear active effects that saved
    return effect_active, effect


def step(state: State, action: Array):
    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    source: Character = jax.tree.map(lambda x: x[*action.source], state.character)
    weapon: ActionArray = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[*action.target], state.character)

    source_ability_bonus = source.attack_ability_mods[weapon.hitroll_type]

    # hitroll for action
    hitroll = 20 - target.ac + source.prof_bonus + source_ability_bonus
    hitroll = jnp.float16(hitroll).clip(1, 20) / 20
    hitroll_mul = jnp.where(weapon.req_hitroll, hitroll, jnp.ones_like(hitroll))

    # saving throw for action
    save_dc = 8 + source.prof_bonus + source_ability_bonus
    save_dc = jnp.where(weapon.use_save_dc, weapon.save_dc, save_dc)
    weapon.save_dc = save_dc  # remember for subsequent saves

    save_fail_prob = save(weapon.save, save_dc, target)
    save_mul = save_fail_prob + (1 - save_fail_prob) * weapon.save_mod
    save_mul = jnp.where(weapon.can_save, save_mul, jnp.ones_like(save_mul))
    recurring_dmg_save_mul = save_fail_prob + (1 - save_fail_prob) * weapon.recurring_damage_save_mod
    recurring_dmg_save_mul = jnp.where(weapon.can_save, recurring_dmg_save_mul, jnp.ones_like(recurring_dmg_save_mul))

    # damage from action
    damage = weapon.damage + jnp.where(weapon.ability_mod_damage, source_ability_bonus, 0)
    damage = damage * hitroll_mul * save_mul * target.damage_type_mul[weapon.damage_type]
    state.character.hp = state.character.hp.at[*action.target].subtract(damage)

    # expectation of recurring damage is reduced if you didn't hit
    weapon.recurring_damage = weapon.recurring_damage * hitroll_mul * recurring_dmg_save_mul * target.damage_type_mul[
        weapon.damage_type]

    # save_fail > 0.5 means the saving throw failed, so set a condition if required
    condition = jnp.where(weapon.inflicts_condition, save_fail_prob > 0.5, False)
    state.character.conditions = state.character.conditions.at[*action.target, weapon.condition].set(condition)
    weapon.cum_save = save_fail_prob

    # add the effect to the effect index if it's going to carry over multiple turns
    effect_index = jnp.argmin(target.effect_active)
    # chex.assert_scalar_in(effect_index, 0, constants.N_EFFECTS)
    effect_active = weapon.duration > 0
    add_effect_if_duration = lambda e, w: jnp.where(effect_active, e.at[effect_index].set(w), e)
    target.effects = jax.tree.map(add_effect_if_duration, target.effects, weapon)
    state.character.effect_active = state.character.effect_active.at[*action.target, effect_index].set(effect_active)
    state.character.effects = jax.tree.map(lambda s, t: s.at[*action.target].set(t), state.character.effects,target.effects)

    prev_effect_active = state.character.effect_active[*action.source]
    effects: ActionArray = jax.tree.map(lambda s: s[*action.source], state.character.effects)
    effect_active, effects = jax.vmap(step_effect, in_axes=(0, None))(effects, source)

    # apply effects to character (map reduce pattern)
    hp = state.character.hp[*action.source] - jnp.sum(effects.recurring_damage * prev_effect_active)

    # I'm sure the below reduce could be optimized to use less memory if I thought about it more
    effect_conditions = jax.nn.one_hot(effects.condition, num_classes=len(constants.Conditions), dtype=jnp.bool)
    effect_conditions = effect_conditions & effect_active.reshape(constants.N_EFFECTS, 1)
    conditions = effect_conditions.any(0)

    end_turn = action.action == Actions['end-turn']

    # update effects and
    state.character.effect_active = update_character_if(action.source, end_turn, state.character.effect_active, effect_active)
    state.character.hp = update_character_if(action.source, end_turn, state.character.hp, hp)
    state.character.conditions = update_character_if(action.source, end_turn, state.character.conditions, conditions)
    state.character.effects = jax.tree.map(partial(update_character_if, action.source, end_turn), state.character.effects, effects)

    # conditions = jnp.where(end_turn, conditions, state.character.conditions[*action.source])

    if debug:
        jax.debug.callback(print_step, source, target, weapon, damage, (1 - save_fail_prob))

    return state
