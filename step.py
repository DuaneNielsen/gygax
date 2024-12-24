import jax
import jax.numpy as jnp
from typing import Dict
from pgx import EnvId

import conditions
from character import CharacterExtra, stack_party, Character
from actions import ActionArray, action_table
from constants import HitrollType, N_PLAYERS, N_CHARACTERS, Abilities
from conditions import map_reduce, hitroll_adv_dis, ConditionModArray, reduce_damage_resist_vun
from dnd5e import ActionTuple, decode_action, Actions
from pgx.core import Array
from to_jax import JaxStringArray
import numpy as np
from functools import partial
from dice import cdf_20, RollType, ad_rule

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


def step_to_str(state: State, action: ActionTuple, damage: jnp.float16,
                save_fail: jnp.float16):
    source: Character = jax.tree.map(lambda x: x[*action.source], state.character)
    weapon: ActionArray = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[*action.target], state.character)
    source_name = JaxStringArray.uint8_array_to_str(source.name)
    target_name = JaxStringArray.uint8_array_to_str(target.name)
    action_name = JaxStringArray.uint8_array_to_str(weapon.name)
    save_fail = np.array(save_fail)
    damage = np.array(damage)
    hitroll_type = weapon.hitroll_type.item()

    save_msg = f' saved {save_fail:.2f}' if weapon.can_save else ''
    concentration_msg = f' {target_name} is concentrating' if target.concentrating.any() else ''

    if hitroll_type in {HitrollType.MELEE, HitrollType.FINESSE}:
        return f'{source_name} hit {target_name} with {action_name} for {damage:.2f}' + save_msg + concentration_msg
    elif hitroll_type == HitrollType.SPELL:
        return f'{source_name} cast {action_name} on {target_name} for {damage:.2f}' + save_msg + concentration_msg
    elif hitroll_type == HitrollType.RANGED:
        return f'{source_name} shot {target_name} with {action_name} for {damage:.2f}' + save_msg + concentration_msg


def print_step(*args):
    print(step_to_str(*args))


def save_fail(save, save_dc, target, roll_type=RollType.NORMAL):
    """
    Probability of target failing the saving throw
    Args:
        save: Ability to use for the save, (CON, WIS etc)
        save_dc: the dc to save against
        target: the character making the save

    Returns: the probability that the save will fail (0..1)

    """
    target_bonus_save = target.prof_bonus * target.save_bonus[save] + target.ability_mods[save]
    save_hurdle = save_dc - target_bonus_save
    return cdf_20(save_hurdle - 1, roll_type)


vmap_save_fail = jax.vmap(save_fail, in_axes=(0, 0, None))


# @chex.chexify
# @jax.jit
# def check_overflow(effect_index, target):
#     #  assert effect_index < target.effect_active.shape[-1]  # check for overflow
#     chex.assert_equal(effect_index < target.effect_active.shape[-1], True)
#     return effect_index


def update_character_if(character, cond, leaf, value):
    updated = leaf.at[*character].set(value)
    return jnp.where(cond, updated, leaf)


def step_effect(effect: ActionArray, character: Character, condition_mods: ConditionModArray):
    """
    Steps the effects on a character 1 step forward
      duration -- duration is reduced by 1 round each step, effect is deactivated when duration is 0
      saving throws -- saving throws over time are accumulate to cancel the effect, ie: save_fail_prob ** steps
      recurring_damage -- recurrent damage amount should be precomputed to account for saving throws and hitroll,
        then it can be simply applied as long as the duration is active

    Args:
        effect: ActionArray (N_EFFECTS, ...)
        character: Character, pytree representing a single character

    Returns: effect active: bool (N_EFFECTS), effect: ActionArray (N_EFFECTS)

    """
    # saving throws to resist active effects
    effect_save_fail_prob = save_fail(effect.save, effect.save_dc, character, roll_type=condition_mods.saving_throw[effect.save])
    effect_save_fail_prob = jnp.where(condition_mods.saving_throw_fail[effect.save], 1., effect_save_fail_prob)
    effect.cum_save = effect.cum_save * effect_save_fail_prob
    effect.cum_save = jnp.where(effect.cum_save > 0.5, effect.cum_save, jnp.zeros_like(effect.cum_save))
    effect.inflicts_condition = jnp.where(effect.inflicts_condition, effect.cum_save > 0.5, False)

    # finally decrease the duration by 1 round
    effect.duration = effect.duration - 1
    effect_active = effect.duration > 0
    effect_active = jnp.where(effect.can_save, effect.cum_save > 0.5, effect_active)  # clear active effects that saved
    return effect_active, effect


def break_concentration(state, concentration_broken, character):
    """
    Breaks concentration of the character and cancels and updates spell effects

    Args:
        state:
        concentration_broken: if true, will cancel the effects of the concentration
        character:

    Returns:

    """
    concentration_ref = state.character.concentration_ref[*character]
    state.character.concentrating = update_character_if(character, concentration_broken, state.character.concentrating,
                                                        False)
    effect_deactivated = state.character.effect_active.at[concentration_ref].set(False)
    state.character.effect_active = jnp.where(concentration_broken, effect_deactivated, state.character.effect_active)
    return state


def update_conditions(effect_active, effects):
    """
    Given the active effects on a character, returns the conditions
    Args:
        effect_active: boolean indicating that the effect is active (..., num_effects)
        effects: indices enumerating effects (..., num_effects)

    Returns: (..., num_conditions)

    """
    # I'm sure the below reduce could be optimized to use less memory if I thought about it more
    effect_conditions = jax.nn.one_hot(effects.condition, num_classes=len(conditions.Conditions), dtype=jnp.bool)
    effect_conditions = effect_conditions & effect_active[..., None]
    return effect_conditions.any(-2)


def hitroll(source: Character, target: Character, weapon: ActionArray, roll_type: RollType = RollType.NORMAL,
            crit: int = 20, auto_crit=False):
    """

    Args:
        source: the attacking character
        target: the target character
        weapon: the action used
        roll_type: NORMAL | ADVANTAGE | DISADVANTAGE
        crit: rolling equal  or higher is considered a crit for double damage, default 20
        auto_crit: any hit automatically crits

    Returns: hit_chance -> probability of hit
             hit_dmg -> multiplier for expected damage
             (hit_prob, crit_prob) crit_prob is the chance to crit, hit prob is the chance to hit normal,
             hit_prob + crit_prob = hit_chance

    """
    source_ability_bonus = source.attack_ability_mods[weapon.hitroll_type]
    effective_ac = target.ac - source.prof_bonus - source_ability_bonus
    effective_ac = jnp.max(jnp.array([effective_ac, 1]))
    hit_prob = cdf_20(crit - 1, roll_type) - cdf_20(effective_ac-1, roll_type)
    crit_prob = 1. - cdf_20(crit - 1, roll_type)
    hit_chance = hit_prob + crit_prob
    hit_prob = jnp.where(auto_crit, 0, hit_prob)
    crit_prob = jnp.where(auto_crit, hit_chance, crit_prob)
    hit_dmg = hit_prob + crit_prob * 2
    return hit_chance, hit_dmg, (hit_prob, crit_prob)


def step(state: State, action: Array):
    """

    Args:
        state:
        action:

    Returns:

    """

    """
    Setup variables
    """

    action = decode_action(action, state.current_player, state.pos, n_actions=len(Actions))
    source: Character = jax.tree.map(lambda x: x[*action.source], state.character)
    weapon: ActionArray = jax.tree.map(lambda action_items: action_items[action.action], action_table)
    target: Character = jax.tree.map(lambda x: x[*action.target], state.character)

    source_ability_bonus = source.attack_ability_mods[weapon.hitroll_type]

    # if the action requires concentration, cancel existing concentration effects
    state = break_concentration(state, weapon.req_concentration & source.concentrating.any(-1), action.source)
    state.character.conditions = update_conditions(state.character.effect_active, state.character.effects)

    # initialize condition modifiers
    source_condition_mods = map_reduce(source.conditions)
    target_condition_mods = map_reduce(target.conditions)
    hitroll_condition_mods = hitroll_adv_dis(source_condition_mods, target_condition_mods)
    melee_hitroll_mod = ad_rule(jnp.stack([hitroll_condition_mods, target_condition_mods.melee_target_hitroll]))
    is_melee_attack = weapon.hitroll_type == HitrollType.MELEE
    hitroll_condition_mods = jnp.where(is_melee_attack, melee_hitroll_mod, hitroll_condition_mods)
    auto_crit = is_melee_attack & target_condition_mods.melee_target_auto_crit

    """
    Action d_Character
    """

    # hit_prob is the chance of a normal hit, crit_prob chance of crit, chance of hit is the sum of the two
    hit_chance, hit_dmg, _ = hitroll(source, target, weapon, roll_type=hitroll_condition_mods, auto_crit=auto_crit)
    hit_dmg = jnp.where(weapon.req_hitroll, hit_dmg, 1.)
    weapon.recurring_damage_hitroll = hit_chance

    # saving throw for action
    save_dc = 8 + source.prof_bonus + source_ability_bonus
    save_dc = jnp.where(weapon.use_save_dc, weapon.save_dc, save_dc)
    weapon.save_dc = save_dc  # remember for subsequent saves

    save_fail_prob = save_fail(weapon.save, save_dc, target, roll_type=target_condition_mods.saving_throw[weapon.save])
    save_fail_prob = jnp.where(target_condition_mods.saving_throw_fail[weapon.save], 1., save_fail_prob)
    save_mul = save_fail_prob + (1 - save_fail_prob) * weapon.save_mod
    save_mul = jnp.where(weapon.can_save, save_mul, jnp.ones_like(save_mul))
    recurring_dmg_save_mul = save_fail_prob + (1 - save_fail_prob) * weapon.recurring_damage_save_mod
    recurring_dmg_save_mul = jnp.where(weapon.can_save, recurring_dmg_save_mul, jnp.ones_like(recurring_dmg_save_mul))

    # damage from action
    damage = weapon.damage + jnp.where(weapon.ability_mod_damage, source_ability_bonus, 0)
    damage_resist_vun = reduce_damage_resist_vun(jnp.stack(
        [target.damage_type_mul[weapon.damage_type], target_condition_mods.damage_resistance[weapon.damage_type]]))
    damage = damage * hit_dmg * save_mul * damage_resist_vun

    state.character.hp = state.character.hp.at[*action.target].subtract(damage)

    # target makes concentration check on hit
    check_concentration_on_hit = target.concentrating.any(-1) & (damage > 0)
    concentration_fail = save_fail(Abilities.CON, 10, target, target_condition_mods.saving_throw[weapon.save])
    concentration_fail = jnp.where(target_condition_mods.saving_throw_fail[weapon.save], 1., concentration_fail)
    concentration_check_cum = (1 - hit_chance * concentration_fail) * target.concentration_check_cum
    concentration_check_fail = check_concentration_on_hit & (concentration_check_cum < 0.5)
    state = break_concentration(state, concentration_check_fail, action.target)
    state.character.concentration_check_cum = update_character_if(action.target, check_concentration_on_hit,
                                                                  state.character.concentration_check_cum,
                                                                  concentration_check_cum)
    state.character.conditions = update_conditions(state.character.effect_active, state.character.effects)

    # expectation of recurring damage is reduced if you didn't hit
    weapon.recurring_damage = weapon.recurring_damage * hit_dmg * recurring_dmg_save_mul * target.damage_type_mul[
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
    state.character.effects = jax.tree.map(lambda s, t: s.at[*action.target].set(t), state.character.effects,
                                           target.effects)

    # update concentration if spell requires it
    state.character.concentrating = update_character_if(action.source, weapon.req_concentration,
                                                        state.character.concentrating, True)
    state.character.concentration_ref = update_character_if(action.source, weapon.req_concentration,
                                                            state.character.concentration_ref, [*action.target, 0])
    state.character.concentration_check_cum = update_character_if(action.source, weapon.req_concentration,
                                                                  state.character.concentration_check_cum, 1.)

    """
    process effects on a character that occur on end-turn
    """
    end_turn = action.action == Actions['end-turn']

    prev_effect_active = state.character.effect_active[*action.source]
    effects: ActionArray = jax.tree.map(lambda s: s[*action.source], state.character.effects)
    effect_active, effects = jax.vmap(step_effect, in_axes=(0, None, None))(effects, source, source_condition_mods)

    # reduce effects to a change in character state
    conditions = update_conditions(effect_active, effects)
    hp = state.character.hp[*action.source] - jnp.sum(effects.recurring_damage * prev_effect_active)

    # update character state
    state.character.hp = update_character_if(action.source, end_turn, state.character.hp, hp)
    state.character.conditions = jnp.where(end_turn, conditions, state.character.conditions)
    state.character.effect_active = update_character_if(action.source, end_turn, state.character.effect_active,
                                                        effect_active)
    state.character.effects = jax.tree.map(partial(update_character_if, action.source, end_turn),
                                           state.character.effects, effects)

    # if effects caused damage, make concentration checks
    damaging_effects = (effects.recurring_damage > 0.) & prev_effect_active
    concentration_fail_prob = save_fail(Abilities.CON, 10, source, source_condition_mods.saving_throw[Abilities.CON])
    concentration_check_cum = jnp.where(damaging_effects,
                                        1 - effects.recurring_damage_hitroll * concentration_fail_prob, 1.).prod()
    concentration_check_cum = concentration_check_cum * source.concentration_check_cum
    state.character.concentration_check_cum = update_character_if(action.source, target.concentrating.any() & end_turn,
                                                                  state.character.concentration_check_cum,
                                                                  concentration_check_cum)
    concentration_check_fail = source.concentrating.any(-1) & damaging_effects.any() & (
            concentration_check_cum < 0.5) & end_turn
    state.character.concentrating = update_character_if(action.source, concentration_check_fail,
                                                        state.character.concentrating,
                                                        False)
    concentration_ref = state.character.concentration_ref[*action.source]
    effect_deactivated = state.character.effect_active.at[concentration_ref].set(False)
    state.character.effect_active = jnp.where(concentration_check_fail, effect_deactivated,
                                              state.character.effect_active)
    state.character.conditions = update_conditions(state.character.effect_active, state.character.effects)

    if debug:
        jax.debug.callback(print_step, state, action, damage, (1 - save_fail_prob))

    return state
