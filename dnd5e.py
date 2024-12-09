# public modules

import jax
import jax.nn as nn
from pgx.core import Array
import pgx.core as core
import chex

import action_resources
from action_resources import legal_actions_by_action_resource, legal_actions_by_player_position
from constants import *
import constants
import turn_tracker
import dice
from equipment.equipment import Equipment, EquipmentType
import equipment.armor as armor
import equipment.weapons as weapons
from collections import namedtuple
from tree_serialization import convert_to_observation
from default_config import default_config
from wrappers import DND5EProxy
from character import CharacterObservation, WeaponArray

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


def _legal_actions(state):
    legal_actions = jnp.ones((N_PLAYERS, N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS), dtype=jnp.bool)
    legal_actions = legal_actions & legal_actions_by_action_resource(state.action_resources.current)
    legal_actions = legal_actions & state.turn_tracker.characters_acting[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    legal_actions = legal_actions & ~state.turn_tracker.end_turn[..., None, None, None]
    legal_actions = legal_actions & legal_actions_by_player_position(state.pos)
    legal_actions = legal_actions & action_resources.legal_target_pos

    return legal_actions[state.current_player]


def encode_action(action, source_character, target_party, target_slot, n_actions=None):
    n_actions = N_ACTIONS if n_actions is None else n_actions
    multi_index = [source_character, action, target_party, target_slot]
    return jnp.ravel_multi_index(multi_index, [N_CHARACTERS, n_actions, N_PLAYERS, N_CHARACTERS])


def decode_action(encoded_action, current_player, pos, n_actions=None):
    n_actions = N_ACTIONS if n_actions is None else n_actions
    source_character, action, target_party, target_slot = jnp.unravel_index(
        encoded_action,[N_CHARACTERS, n_actions, N_PLAYERS,N_CHARACTERS])

    # reverse the target party for NPCs
    target_party = (target_party + current_player) % N_PLAYERS
    target_character = pos[target_party, target_slot]
    return ActionTuple(Character(current_player, source_character), action, Character(target_party, target_character),
                       CharacterSlot(target_party, target_slot))


# @chex.dataclass
# class Party:
#     pos: chex.ArrayDevice
#     hitpoints: chex.ArrayDevice  # hit points
#     hitpoints_max: chex.ArrayDevice  # characters max hitpoints
#     armor_class: chex.ArrayDevice  # armor class
#     proficiency_bonus: chex.ArrayDevice  # proficiency bonus
#     ability_modifier: chex.ArrayDevice  # ability bonus for each stat
#     class_ability_bonus_idx: chex.ArrayDevice  # class ability index 0: STR, 1: DEX, 2: CON,
#     action_resources_start_turn: chex.ArrayDevice
#     action_resources: chex.ArrayDevice  # number of actions remaining
#     conditions: chex.ArrayDevice  # condition stacks


# @chex.dataclass
# class ObservationParty:
#     # pos: chex.ArrayDevice
#     hitpoints: chex.ArrayDevice  # hit points
#     armor_class: chex.ArrayDevice  # armor class
#     proficiency_bonus: chex.ArrayDevice  # proficiency bonus
#     ability_modifier: chex.ArrayDevice  # ability bonus for each stat
#     class_ability_bonus: chex.ArrayDevice  # class ability index 0: STR, 1: DEX, 2: CON,
#     action_resources: chex.ArrayDevice  # number of actions remaining
#     conditions: chex.ArrayDevice  # condition stacks


# def init_party():
#     return Party(
#         pos=jnp.arange(N_CHARACTERS).repeat(N_PLAYERS).reshape(N_CHARACTERS, N_PLAYERS).T,
#         hitpoints=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.float32),
#         hitpoints_max=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.float32),
#         armor_class=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),
#         proficiency_bonus=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),  # proficiency bonus
#         ability_modifier=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ABILITIES), dtype=jnp.int32),
#         class_ability_bonus_idx=jnp.zeros((N_PLAYERS, N_CHARACTERS), dtype=jnp.int32),
#         action_resources_start_turn=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32),
#         action_resources=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32),
#         conditions=jnp.zeros((N_PLAYERS, N_CHARACTERS, N_CONDITIONS), dtype=jnp.int32)
#     )


# def _observe_party(party: Party):
#     return ObservationParty(
#         hitpoints=cum_bins(party.hitpoints, HP_UPPER, HP_LOWER),
#         armor_class=cum_bins(party.armor_class, AC_UPPER, AC_LOWER),
#         proficiency_bonus=cum_bins(party.proficiency_bonus, PROF_BONUS_UPPER, PROF_BONUS_LOWER),
#         ability_modifier=cum_bins(party.ability_modifier, ABILITY_MODIFIER_UPPER, ABILITY_MODIFIER_LOWER),
#         class_ability_bonus=nn.one_hot(party.class_ability_bonus_idx, N_ABILITIES),
#         conditions=cum_bins(party.conditions, CONDITION_STACKS_UPPER),
#         action_resources=cum_bins(party.action_resources, ACTION_RESOURCES_UPPER),
#     )


# @chex.dataclass
# class Observation:
#     party: ObservationParty


from pgx._src.struct import dataclass
from character import CharacterArray, stack_party
from action_resources import ActionResourceArray
from functools import partial


@dataclass
class State:
    names: list
    # dnd5e specific
    party: CharacterArray
    turn_tracker: turn_tracker.TurnTracker
    pos: jnp.int8
    action_resources: jnp.int8

    observation: CharacterObservation
    legal_action_mask: Array = jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS))
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _step_count: Array = jnp.zeros((1,), dtype=jnp.int32)

    @property
    def current_player(self):
        return self.turn_tracker.party.squeeze(0)

    @current_player.setter
    def current_player(self, value):
        self.turn_tracker.party = jnp.int32(value)

    @property
    def env_id(self) -> core.EnvId:
        return "5e_srd_dd_style"


def _init(rng: jax.random.PRNGKey, config) -> State:
    names, party = stack_party(config[ConfigItems.PARTY])
    pos = jnp.tile(jnp.arange(N_CHARACTERS, dtype=jnp.int8), reps=(N_PLAYERS, 1))
    tt = turn_tracker.init(party.ability_modifier.dexterity)
    action_resources = jax.tree.map(partial(jnp.tile, reps=(N_PLAYERS, N_CHARACTERS)), ActionResourceArray())

    state = State(
        names=names,
        party=party,
        pos=pos,
        turn_tracker=tt,
        action_resources=action_resources,
        observation=None,
    )
    observation = _observe(state, state.current_player)
    return state.replace(
        observation=observation,
        legal_action_mask=_legal_actions(state).ravel()
    )


def _observe(state: State, player_id: Array) -> Array:
    R_PLAYER = (jnp.arange(N_PLAYERS) + player_id) % 2
    R_PLAYER = R_PLAYER[..., None]

    # permute party positions to create egocentric view
    party = jax.tree_map(lambda party: party[R_PLAYER, state.pos], state.party)
    return convert_to_observation(party, CharacterObservation)


def _win_check(state):
    party_killed = jnp.all(state.party.dead, axis=1)
    return jnp.any(party_killed), (jnp.argmax(party_killed) + 1) % N_PLAYERS


def end_turn(state, action):
    # jax.debug.print('state.scene.turn_tracker.characters_acting \n{}', state.scene.turn_tracker.characters_acting)
    state.turn_tracker = turn_tracker.next_turn(state.turn_tracker,
                                                action.action == Actions.END_TURN,
                                                action.source.party, action.source.index)
    return state


def print_damage(char_names, amount, party, index):
    target = char_names[party.item(), index.item()]
    print(f'damage {amount.item()} {target}')


def apply_damage(state: State, target: Character, weapon: WeaponArray):
    new_hp = state.party.current_hp[*target] - weapon.expected_damage
    state.party.hitpoints = state.party.current_hp.at[*target].set(new_hp)
    if debug:
        jax.debug.callback(print_damage, state.names, weapon.expected_damage, target.party, target.index)
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
    action = decode_action(action, state.current_player, state.pos)
    state.action_resources.current = action_resources.consume_action_resource(state.action_resources.current, action)
    if debug:
        jax.debug.callback(print_action, action)
    state = end_turn(state, action)
    # jax.debug.print('on_turn_start {}', state.scene.turn_tracker.on_turn_start)
    state.action_resources.current = jnp.where(state.turn_tracker.on_turn_start,
                                               state.action_resources.start_turn,
                                               state.action_resources.current)

    # actions that take effect on the turn start occur before this line
    state.scene.turn_tracker = turn_tracker.clear_events(state.scene.turn_tracker)

    weapon = jax.tree.map(jnp.zeros_like, state.party.main_attack)
    weapon = jnp.where(action == Actions.ATTACK_MELEE_WEAPON, state.party.main_attack, weapon)
    weapon = jnp.where(action == Actions.ATTACK_RANGED_WEAPON, state.party.main_attack, weapon)
    weapon = jnp.where(action == Actions.ATTACK_OFF_HAND_WEAPON, state.party.main_attack, weapon)

    state = apply_damage(state, action.target, weapon)

    state = apply_death(state)

    game_over, winner = _win_check(state)

    reward = jax.lax.cond(
        game_over,
        lambda: jnp.float32([-1, -1]).at[winner].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    legal_action_mask = _legal_actions(state)

    return state.replace(
        legal_action_mask=legal_action_mask.ravel(),  # ravel flattens the action_mask
        rewards=reward,
        terminated=game_over
    )


class DND5E(core.Env):
    def __init__(self, config=None):
        super().__init__()
        config = default_config if config is None else config
        self._init_state: State = _init(None, config)

    def init(self, key: jax.random.PRNGKey) -> State:
        """
        This method is overridden so we can return the correct type hint
        """
        return super().init(key)

    def step(self, state: State, action: int, key: jax.random.PRNGKey = None) -> State:
        """
        This method is overridden so we can return the correct type hint
        """
        return super().step(state, action, key)

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
        return "gygax-5_1"

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
