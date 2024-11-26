# public modules

import jax
import jax.numpy as jnp
import jax.nn as nn
from pgx.core import Array
import pgx.core as core
import chex

from constants import *
import constants
import turn_tracker
import dice
from equipment.equipment import Equipment, EquipmentType
import equipment.armor as armor
import equipment.weapons as weapons
from collections import namedtuple
from tree_serialization import cum_bins
from default_config import default_config
from wrappers import DND5EProxy

"""
This simulation is deterministic.
Any time a dice is rolled, we take the expected value

The way to think about it is, if we were to run the random process a near infinite amount of times,
what would be the result on average (the expectation)

For conditional process, we use the conditional probability 
eg: to calculate damage from a melee attack, we multiply the hit probability by the expected damage roll
This simplifies things a lot, and significantly reduces the amount of computation required

It also allows us to use alphazero without needing to add "chance nodes" for stochastic alphazero,
which would significantly increase the amount of resources required to traverse the game tree
"""

# CharacterAction = namedtuple('Character', ['source_party', 'source_character', ])
Damage = namedtuple('Damage', ['amount', 'type'])
Character = namedtuple('Character', ['party', 'index'])
CharacterSlot = namedtuple('Character', ['party', 'slot'])
ActionTuple = namedtuple('ActionTuple', ['source', 'action', 'target', 'target_slot'])


def legal_actions_by_action_resource(action_resources):
    legal_actions = jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.bool)

    # end turn does not require resources
    legal_actions = legal_actions.at[:, :, Actions.END_TURN].set(TRUE)

    # weapons require an action or an attack resource
    actions = action_resources[:, :, ActionResourceType.ACTION] > 0
    attacks = action_resources[:, :, ActionResourceType.ATTACK] > 0
    legal_actions = legal_actions.at[:, :, Actions.ATTACK_MELEE_WEAPON].set(actions | attacks)
    legal_actions = legal_actions.at[:, :, Actions.ATTACK_RANGED_WEAPON].set(actions | attacks)
    return legal_actions[..., jnp.newaxis, jnp.newaxis]


def legal_actions_by_player_position(pos, legal_use_pos):
    """
    True if character is in position to perform action
    :param pos: (Party, Characters)
    :param legal_use_pos: (Party, Characters, Action, Position)
    :return: boolean (Party, Characters, Action, 1, 1)
    """
    leading_idx = jnp.indices(legal_use_pos.shape[0:3])
    POS = pos[:, :, None]

    legal_pos_for_action = legal_use_pos[*leading_idx, POS]
    return legal_pos_for_action[..., jnp.newaxis, jnp.newaxis]


def _legal_actions(scene, current_player):
    legal_actions = jnp.ones((N_PLAYERS, N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS), dtype=jnp.bool)
    legal_actions = legal_actions & legal_actions_by_action_resource(scene.party.action_resources)
    legal_actions = legal_actions & scene.turn_tracker.characters_acting[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    legal_actions = legal_actions & ~scene.turn_tracker.end_turn[..., None, None, None]
    legal_actions = legal_actions & legal_actions_by_player_position(scene.party.pos, scene.party.actions.legal_use_pos)
    legal_actions = legal_actions & scene.party.actions.legal_target_pos

    return legal_actions[current_player]


def encode_action(action, source_character, target_party, target_slot):
    multi_index = [source_character, action, target_party, target_slot]
    return jnp.ravel_multi_index(multi_index, [N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS])


def decode_action(encoded_action, current_player, pos):
    source_character, action, target_party, target_slot = jnp.unravel_index(encoded_action,
                                                                            [N_CHARACTERS, N_ACTIONS, N_PLAYERS,
                                                                             N_CHARACTERS])
    # reverse the target party for NPCs
    target_party = (target_party + current_player) % N_PLAYERS
    target_character = pos[target_party, target_slot]
    return ActionTuple(Character(current_player, source_character), action, Character(target_party, target_character),
                       CharacterSlot(target_party, target_slot))


@chex.dataclass
class ActionArray:
    type: chex.ArrayDevice
    damage: chex.ArrayDevice
    damage_type: chex.ArrayDevice
    legal_use_pos: chex.ArrayDevice
    legal_target_pos: chex.ArrayDevice
    resource_requirement: chex.ArrayDevice


@chex.dataclass
class ObservationActionArray:
    type: chex.ArrayDevice
    damage: chex.ArrayDevice
    damage_type: chex.ArrayDevice
    legal_use_pos: chex.ArrayDevice
    legal_target_pos: chex.ArrayDevice
    resource_requirement: chex.ArrayDevice


def init_actions():
    return ActionArray(
        type=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.int32),
        damage=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.float32),
        damage_type=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.int32),
        legal_use_pos=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS, N_CHARACTERS), dtype=jnp.bool),
        legal_target_pos=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS), dtype=jnp.bool),
        resource_requirement=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS), dtype=jnp.int32),
    )


def _observe_actions(actions: ActionArray):
    return ObservationActionArray(
        type=nn.one_hot(actions.type, N_ACTIONS),
        damage=cum_bins(jnp.int32(actions.damage), DAMAGE_UPPER),
        damage_type=nn.one_hot(actions.damage_type, N_DAMAGE_TYPES),
        legal_use_pos=jnp.float32(actions.legal_use_pos),
        legal_target_pos=jnp.float32(actions.legal_target_pos),
        resource_requirement=jnp.float32(actions.resource_requirement),
    )


def convert_equipment(item: Equipment):
    if isinstance(item, weapons.Weapon):
        if item.type == EquipmentType.WEAPON_MELEE:
            return ActionArray(
                type=jnp.array(item.type, dtype=jnp.int32),
                damage=jnp.array(dice.expected_roll(item.damage), dtype=jnp.float32),
                damage_type=jnp.array(item.damage_type, dtype=jnp.int32),
                legal_use_pos=jnp.array([0, 0, 1, 1], dtype=jnp.bool),
                legal_target_pos=jnp.array([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=jnp.bool),
                resource_requirement=jnp.array(ActionResourceType.ATTACK, dtype=jnp.int32),
            )
        if item.type == EquipmentType.WEAPON_RANGED:
            return ActionArray(
                type=jnp.array(item.type, dtype=jnp.int32),
                damage=jnp.array(dice.expected_roll(item.damage), dtype=jnp.float32),
                damage_type=jnp.array(item.damage_type, dtype=jnp.int32),
                legal_use_pos=jnp.array([1, 1, 0, 0], dtype=jnp.bool),
                legal_target_pos=jnp.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=jnp.bool),
                resource_requirement=jnp.array(ActionResourceType.ATTACK, dtype=jnp.int32),
            )


@chex.dataclass
class Party:
    pos: chex.ArrayDevice
    hitpoints: chex.ArrayDevice  # hit points
    hitpoints_max: chex.ArrayDevice  # characters max hitpoints
    armor_class: chex.ArrayDevice  # armor class
    proficiency_bonus: chex.ArrayDevice  # proficiency bonus
    ability_modifier: chex.ArrayDevice  # ability bonus for each stat
    class_ability_bonus_idx: chex.ArrayDevice  # class ability index 0: STR, 1: DEX, 2: CON,
    actions: ActionArray  # the characters equipment
    action_resources_start_turn: chex.ArrayDevice
    action_resources: chex.ArrayDevice  # number of actions remaining
    conditions: chex.ArrayDevice  # condition stacks


@chex.dataclass
class ObservationParty:
    # pos: chex.ArrayDevice
    hitpoints: chex.ArrayDevice  # hit points
    armor_class: chex.ArrayDevice  # armor class
    proficiency_bonus: chex.ArrayDevice  # proficiency bonus
    ability_modifier: chex.ArrayDevice  # ability bonus for each stat
    class_ability_bonus: chex.ArrayDevice  # class ability index 0: STR, 1: DEX, 2: CON,
    actions: ObservationActionArray  # the characters equipment
    action_resources: chex.ArrayDevice  # number of actions remaining
    conditions: chex.ArrayDevice  # condition stacks


def init_party():
    return Party(
        pos=jnp.arange(N_CHARACTERS).repeat(N_PLAYERS).reshape(N_CHARACTERS, N_PLAYERS).T,
        hitpoints=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.float32),
        hitpoints_max=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.float32),
        armor_class=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),
        proficiency_bonus=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),  # proficiency bonus
        ability_modifier=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ABILITIES), dtype=jnp.int32),
        class_ability_bonus_idx=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),
        actions=init_actions(),
        action_resources_start_turn=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32),
        action_resources=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32),
        conditions=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_CONDITIONS), dtype=jnp.int32)
    )


def _observe_party(party: Party):
    return ObservationParty(
        hitpoints=cum_bins(party.hitpoints, HP_UPPER, HP_LOWER),
        armor_class=cum_bins(party.armor_class, AC_UPPER, AC_LOWER),
        proficiency_bonus=cum_bins(party.proficiency_bonus, PROF_BONUS_UPPER, PROF_BONUS_LOWER),
        ability_modifier=cum_bins(party.ability_modifier, ABILITY_MODIFIER_UPPER, ABILITY_MODIFIER_LOWER),
        class_ability_bonus=nn.one_hot(party.class_ability_bonus_idx, N_ABILITIES),
        conditions=cum_bins(party.conditions, CONDITION_STACKS_UPPER),
        action_resources=cum_bins(party.action_resources, ACTION_RESOURCES_UPPER),
        actions=_observe_actions(party.actions)
    )


@chex.dataclass
class Scene:
    party: Party
    turn_tracker: turn_tracker.TurnTracker


@chex.dataclass
class Observation:
    party: ObservationParty


def ability_modifier(ability_score):
    return (ability_score - 10) // 2


def configure_party(config):
    party = init_party()
    party_config = config[ConfigItems.PARTY]
    for p in constants.Party:
        for C, (name, character_sheet) in enumerate(party_config[p].items()):
            P = p.value

            # ability scores
            for ability in Abilities:
                ability_score = character_sheet[CharacterSheet.ABILITIES][ability]
                party.ability_modifier = party.ability_modifier.at[P, C, ability.value].set(
                    ability_modifier(ability_score))

            party.hitpoints = party.hitpoints.at[P, C].set(character_sheet[CharacterSheet.HITPOINTS])
            party.hitpoints_max = party.hitpoints.at[P, C].set(character_sheet[CharacterSheet.HITPOINTS])

            # action resources
            party.action_resources_start_turn = party.action_resources_start_turn.at[
                P, C, ActionResourceType.ACTION].set(1)
            party.action_resources_start_turn = party.action_resources_start_turn.at[
                P, C, ActionResourceType.BONUS_ACTION].set(1)
            party.action_resources = jnp.copy(party.action_resources_start_turn)

            # armor class
            dex_ability_modifier = ability_modifier(character_sheet[CharacterSheet.ABILITIES][Abilities.DEX])
            ac_dex_bonus = min(character_sheet[CharacterSheet.ARMOR].max_dex_bonus, dex_ability_modifier)
            has_shield = character_sheet[CharacterSheet.OFF_HAND].type == armor.ArmorType.SHIELD
            armour_class = character_sheet[CharacterSheet.ARMOR].ac + ac_dex_bonus + has_shield * 2
            party.armor_class = party.armor_class.at[P, C].set(armour_class)

            # end turn action
            party.actions.legal_use_pos = party.actions.legal_use_pos.at[:, :, Actions.END_TURN].set(True)
            party.actions.legal_target_pos = party.actions.legal_target_pos.at[:, :, Actions.END_TURN, 0, 0].set(True)

            # melee weapon
            item = convert_equipment(character_sheet[CharacterSheet.MAIN_HAND])
            party.actions = jax.tree.map(lambda x, y: x.at[P, C, Actions.ATTACK_MELEE_WEAPON].set(y), party.actions,
                                         item)

            # ranged weapon
            item = convert_equipment(character_sheet[CharacterSheet.RANGED_WEAPON])
            party.actions = jax.tree.map(lambda x, y: x.at[P, C, Actions.ATTACK_RANGED_WEAPON].set(y), party.actions,
                                         item)
    return party


def init_scene(config=None):
    config = config if config is not None else default_config
    party = configure_party(config)
    dex_ability_bonus = party.ability_modifier[:, :, Abilities.DEX]

    return Scene(
        party=party,
        turn_tracker=turn_tracker.init(dex_ability_bonus=dex_ability_bonus)
    )


from pgx._src.struct import dataclass


@dataclass
class State:
    # dnd5e specific
    scene: Scene
    observation: Observation
    legal_action_mask: Array = jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS))
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _step_count: Array = jnp.zeros((1,), dtype=jnp.int32)

    @property
    def current_player(self):
        return self.scene.turn_tracker.party

    @current_player.setter
    def current_player(self, value):
        self.scene.turn_tracker.party = jnp.int32(value)

    @property
    def env_id(self) -> core.EnvId:
        return "5e_srd_dd_style"


def _init(rng: jax.random.PRNGKey, config) -> State:
    scene = init_scene(config)
    legal_action_mask: Array = _legal_actions(scene, current_player=jnp.array([constants.Party.PC]))

    return State(
        scene=scene,
        observation=Observation(
            party=_observe_party(scene.party)
        ),
        legal_action_mask=legal_action_mask.ravel()
    )


def _observe(state: State, player_id: Array) -> Array:
    R_PLAYER = (jnp.arange(N_PLAYERS) + player_id) % 2
    R_PLAYER = R_PLAYER[..., None]

    # permute party positions to create egocentric view
    party = jax.tree_map(lambda party: party[R_PLAYER, state.scene.party.pos], state.scene.party)
    return Observation(
        party=_observe_party(party)
    )


def _win_check(state):
    party_killed = jnp.all(state.scene.party.conditions[:, :, Conditions.DEAD] > 0, axis=1)
    return jnp.any(party_killed), (jnp.argmax(party_killed) + 1) % N_PLAYERS


def end_turn(state, action):
    # jax.debug.print('state.scene.turn_tracker.characters_acting \n{}', state.scene.turn_tracker.characters_acting)
    state.scene.turn_tracker = turn_tracker.next_turn(state.scene.turn_tracker,
                                                      action.action == Actions.END_TURN,
                                                      action.source.party, action.source.index)
    return state


def print_damage(amount, party, index):
    target = char_names[party.item()][index.item()]
    print(f'damage {amount.item()} {target}')


def apply_damage(state: State, target: Character, damage: Damage):
    new_hp = state.scene.party.hitpoints[*target] - damage.amount
    state.scene.party.hitpoints = state.scene.party.hitpoints.at[*target].set(new_hp)
    if debug:
        jax.debug.callback(print_damage, damage.amount, target.party, target.index)
    return state


def weapon_attack(state, action):
    # deterministic attack
    # target_ac = state.scene.party.armor_class[action.target_party, target_character]
    damage = Damage(
        amount=state.scene.party.actions.damage[*action.source, action.action],
        type=state.scene.party.actions.damage_type[*action.source, action.action])

    # jax.debug.print('source {}, {}, target {}, {} damage {}, {}', *source, *target, *damage)

    state = apply_damage(state, action.target, damage)
    # action_resources = state.scene.party.action_resources.at[*action.source].set(
    #     state.scene.party.action_resources[*action.source] - 1)

    # first use up attacks, else use up an action
    action_resources = state.scene.party.action_resources
    has_attacks = action_resources[:, :, ActionResourceType.ATTACK, None] > 0
    weapon_attacked = (action.action == Actions.ATTACK_MELEE_WEAPON) | (action.action == Actions.ATTACK_RANGED_WEAPON)
    consume_action = action_resources.at[*action.source, ActionResourceType.ACTION].set(
        action_resources[*action.source, ActionResourceType.ACTION] - 1)
    consume_attack = action_resources.at[*action.source, ActionResourceType.ATTACK].set(
        action_resources[*action.source, ActionResourceType.ATTACK] - 1)
    action_resources = jnp.where(weapon_attacked & has_attacks, consume_attack, action_resources)
    state.scene.party.action_resources = jnp.where(weapon_attacked & ~has_attacks, consume_action, action_resources)
    # todo: add bonus attacks here
    return state


def apply_death(state):
    state.scene.party.conditions = state.scene.party.conditions.at[:, :, Conditions.DEAD].set(
        (state.scene.party.hitpoints) <= 0 * 1)
    dead_characters = state.scene.party.conditions[:, :, Conditions.DEAD] > 0
    zero_action_resources = jnp.zeros_like(state.scene.party.action_resources)
    state.scene.party.action_resources = jnp.where(dead_characters[..., None], zero_action_resources,
                                                   state.scene.party.action_resources)
    return state


global char_names


def repr_action(action: ActionTuple):
    def name(char_names, character: Character):
        return char_names[character.party.item()][character.index.item()]

    global char_names
    source = name(char_names, action.source)
    target = name(char_names, action.target)
    return f'{source} {Actions(action.action).name} {target}'


def print_action(action):
    print(repr_action(action))


debug = False


def _step(state: State, action: Array) -> State:
    action = decode_action(action, state.current_player, state.scene.party.pos)
    if debug:
        jax.debug.callback(print_action, action)
    state = end_turn(state, action)
    # jax.debug.print('on_turn_start {}', state.scene.turn_tracker.on_turn_start)
    state.scene.party.action_resources = jnp.where(state.scene.turn_tracker.on_turn_start,
                                                   state.scene.party.action_resources_start_turn,
                                                   state.scene.party.action_resources)
    # jax.debug.print('jimmy action_resources {}', state.scene.party.action_resources[0, 1])

    # actions that take effect on the turn start occur before this line
    state.scene.turn_tracker = turn_tracker.clear_events(state.scene.turn_tracker)

    state = weapon_attack(state, action)

    state = apply_death(state)

    game_over, winner = _win_check(state)

    reward = jax.lax.cond(
        game_over,
        lambda: jnp.float32([-1, -1]).at[winner].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    legal_action_mask = _legal_actions(state.scene, state.current_player)

    return state.replace(
        legal_action_mask=legal_action_mask.ravel(),  # ravel flattens the action_mask
        rewards=reward,
        terminated=game_over
    )


class DND5E(core.Env):
    def __init__(self, config=None):
        super().__init__()
        self._init_state: State = _init(None, config)
        config = default_config if config is None else config
        global char_names
        char_names = [list(config[ConfigItems.PARTY][party].keys()) for party in constants.Party]

    def _init(self, key: jax.random.PRNGKey) -> State:
        del key
        return jax.tree.map(lambda x: x.copy(), self._init_state)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "tic_tac_toe"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


"""
wrappers below here
"""


def wrap_reward_on_hitbar_percentage(env):
    """
    create a dense reward of damage dealt / total party hitpoints
    """

    def new_step(parent_env, state, action, key):
        prev_state = jax.tree.map(lambda s: s.copy(), state)
        next_state = parent_env.step(state, action, key)
        prev_hp = jnp.clip(prev_state.scene.party.hitpoints, min=0)
        next_hp = jnp.clip(next_state.scene.party.hitpoints, min=0)
        reward = prev_hp - next_hp
        reward /= prev_state.scene.party.hitpoints_max
        reward = reward.sum(-1) / N_CHARACTERS
        reward = reward[(jnp.arange(N_PLAYERS) + 1) % 2]
        return next_state.replace(rewards=reward)

    return DND5EProxy(env, step_wrapper=new_step)


def wrap_win_first_death(env):
    def new_step(parent_env, state, action, key):
        winner = state.current_player.squeeze(-1)
        next_state = parent_env.step(state, action, key)
        # game over if anyones hitpoints drops below or equal 0
        game_over = jnp.any(jnp.sum(next_state.scene.party.hitpoints <= 0, -1) > 0)
        reward = jax.lax.cond(
            game_over,
            lambda: jnp.float32([-1, -1]).at[winner].set(1),
            lambda: jnp.zeros(2, jnp.float32),
        )
        return next_state.replace(
            rewards=reward,
            terminated=game_over
        )

    return DND5EProxy(env, step_wrapper=new_step)


def wrap_party_initiative(env, party, initiative_mod):
    def new_init(parent_env, rng_key):
        state = parent_env.init(rng_key)
        init_score = state.scene.turn_tracker.initiative_scores
        new_initiative = init_score.at[party].set(init_score[party] + initiative_mod)
        state.scene.turn_tracker.initiative_scores = new_initiative
        return state

    return DND5EProxy(env, init_wrapper=new_init)