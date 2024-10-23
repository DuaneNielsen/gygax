import jax
import jax.numpy as jnp

import pgx.core as core
from constants import *
import constants
import turn_tracker
import chex
from pgx.core import Array

"""
This simulation is deterministic.
Any time a dice is rolled, we take the expected value

The way to think about it is, if we were to run the random process a near infinite amount of times,
what would be the result on average (the expectation)

For conditional process, we use the conditional probability 
eg: to calculate damage from a melee attack, we multiply the hit probability by the expected damage roll
This simplifies things a lot, and significantly reduces the amount of computation required

It also allows us to use alphazero without needing to add "chance nodes" for stochastic alphazero,
which would significantly complicate things 
"""


def legal_actions_by_action_resource(action_resources):
    legal_actions = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS), dtype=jnp.bool)

    # end turn is always a legal action
    legal_actions = legal_actions.at[:, :, Actions.END_TURN].set(TRUE)
    return legal_actions[..., jnp.newaxis, jnp.newaxis]


def _legal_actions(scene):
    legal_actions = jnp.ones((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS, N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.bool)
    legal_actions = legal_actions & legal_actions_by_action_resource(scene.party.action_resources)
    legal_actions = legal_actions & scene.turn_tracker.characters_acting[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    return legal_actions


def encode_action(action, source_party, source_character, target_party, target_slot):
    multi_index = [source_party, source_character, action, target_party, target_slot]
    return jnp.ravel_multi_index(multi_index, [N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS, N_PLAYERS, MAX_PARTY_SIZE])


def decode_action(encoded_action):
    return jnp.unravel_index(encoded_action, [N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS, N_PLAYERS, MAX_PARTY_SIZE])


@chex.dataclass
class Damage:
    amount: chex.ArrayDevice  # expected damage
    type: chex.ArrayDevice  # damage type


def init_damage():
    return Damage(
        amount=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.float32),
        type=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.int32)
    )


@chex.dataclass
class WeaponSlots:
    legal_use_pos: chex.ArrayDevice
    legal_target_pos: chex.ArrayDevice
    magic_bonus: chex.ArrayDevice
    damage: Damage


def init_weapon_slots():
    return WeaponSlots(
        legal_use_pos=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS, 1, MAX_PARTY_SIZE), dtype=jnp.bool),
        legal_target_pos=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS, N_PLAYERS, MAX_PARTY_SIZE),
                                   dtype=jnp.bool),
        magic_bonus=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.bool),
        damage=init_damage()
    )


@chex.dataclass
class Party:
    pos: chex.ArrayDevice
    hitpoints: chex.ArrayDevice  # hit points
    armor_class: chex.ArrayDevice  # armor class
    proficiency_bonus: chex.ArrayDevice  # proficiency bonus
    ability_modifier: chex.ArrayDevice  # ability bonus for each stat
    class_ability_bonus_idx: chex.ArrayDevice  # class ability index 0: STR, 1: DEX, 2: CON,
    weapons: WeaponSlots
    actions_resources_start_turn: chex.ArrayDevice  # number of actions at the start of the turn
    action_resources: chex.ArrayDevice  # number of actions remaining


def init_party():
    return Party(
        pos=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32),
        hitpoints=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.float32),
        armor_class=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32),
        proficiency_bonus=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32),  # proficiency bonus
        ability_modifier=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ABILITIES), dtype=jnp.int32),
        class_ability_bonus_idx=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32),
        weapons=init_weapon_slots(),
        actions_resources_start_turn=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32),
        action_resources=jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32)
    )


@chex.dataclass
class Scene:
    party: Party
    turn_tracker: turn_tracker.TurnTracker



from default_config import default_config


def init_scene(config=None):
    config = config if config is not None else default_config
    party = init_party()
    party_config = config[ConfigItems.PARTY]
    for p in constants.Party:
        for c, (name, character_sheet) in enumerate(party_config[p].items()):
            for ability in Abilities:
                ability_score = character_sheet[CharacterStats.ABILITIES][ability]
                ability_modifier = (ability_score - 10) // 2
                party.ability_modifier = party.ability_modifier.at[p.value, c, ability.value].set(ability_modifier)

            party.hitpoints = party.hitpoints.at[p.value, c].set(character_sheet[CharacterStats.HITPOINTS])

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
    legal_action_mask: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS))
    observation: Array = jnp.zeros((3, 3, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    _step_count: Array = jnp.zeros((1, ), dtype=jnp.int32)

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
    legal_action_mask: Array = _legal_actions(scene)

    return State(
        scene=scene,
        legal_action_mask=legal_action_mask.ravel()
    )


def _observe(state: State, player_id: Array) -> Array:
    return state.observation


def _win_check(state):
    party_killed = jnp.all(state.scene.party.hitpoints <= 0., axis=1)
    return jnp.any(party_killed), ~jnp.argmax(party_killed)


def end_turn(state, action, source_party, source_character, target_party, target_slot):
    new_tt = turn_tracker.next_turn(state.scene.turn_tracker, source_party, source_character)
    f = lambda new, old: jnp.where(action == Actions.END_TURN, new, old)
    state.scene.turn_tracker = jax.tree_map(f, new_tt, state.scene.turn_tracker)
    return state


def _step(state: State, action: Array) -> State:
    source_party, source_character, action, target_party, target_slot = decode_action(action)
    jax.debug.print('action {} source_party {} source_character {}', action, source_party, source_character)

    # actions that take effect on the turn start occur before this line
    state.scene.turn_tracker = turn_tracker.end_on_character_start(state.scene.turn_tracker)

    state = end_turn(state, action, source_party, source_character, target_party, target_slot)

    game_over, winner = _win_check(state)

    reward = jax.lax.cond(
        game_over,
        lambda: jnp.float32([-1, -1]).at[winner].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )

    legal_action_mask: Array = _legal_actions(state.scene)

    return state.replace(
        legal_action_mask=legal_action_mask.ravel(),  # ravel flattens the action_mask
        rewards=reward,
        terminated=game_over
    )


class DND5E(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.PRNGKey, config=None) -> State:
        return _init(key, config)

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
