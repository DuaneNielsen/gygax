import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from constants import *
import turn_tracker


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

    any_actions_remaining = (action_resources > 0).any(-1)
    legal_actions = legal_actions.at[:, :, Actions.END_TURN].set(any_actions_remaining)
    # action_available = ACTION_RESOURCE_TABLE & action_resources.reshape(1, 1, 4)
    return legal_actions


def _legal_actions(turn_tracker, action_resources):
    legal_actions = jnp.ones((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS), dtype=jnp.bool)
    resources_available = legal_actions_by_action_resource(action_resources)

    legal_actions = legal_actions & turn_tracker.characters_turn.unsqueeze(-1) & resources_available
    return legal_actions





def encode_action(action, source_party, source_character, target_party, target_slot):
    return


def decode_action(encoded_action):
    character = encoded_action % N_ACTIONS
    action = encoded_action - character * N_ACTIONS
    return character, action


@dataclass
class Damage:
    amount: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.float32)  # expected damage
    type: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.int32)  # damage type


@dataclass
class WeaponSlots:
    legal_use_pos: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS, 1, MAX_PARTY_SIZE), dtype=jnp.bool)
    legal_target_pos: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS, N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.bool)
    magic_bonus: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_WEAPON_SLOTS), dtype=jnp.bool)
    damage: Damage = Damage()


@dataclass
class Party:
    hitpoints: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.float32)  # hit points
    armor_class: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)  # armor class
    proficiency_bonus: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)  # proficiency bonus
    ability_bonus: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ABILITIES), dtype=jnp.int32)  # ability bonus for each stat
    class_ability_bonus_idx = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)  # class ability index 0: STR, 1: DEX, 2: CON,
    weapons: WeaponSlots = WeaponSlots()
    actions_start_turn : Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32)  # number of actions at the start of the turn
    actions_remaining : Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTION_RESOURCE_TYPES), dtype=jnp.int32) # number of actions remaining


class Scene:
    def __init__(self, party: Party):
        self.player_pos: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)
        self.party: Party = party
        self.turn_tracker: turn_tracker.init(dex_ability_bonus=party.ability_bonus[:, :, Abilities.DEX])


class State(core.State):

    def __init__(self):
        self.rewards: Array = jnp.float32([0.0, 0.0])
        self.terminated: Array = FALSE
        self.truncated: Array = FALSE
        self._step_count: Array = jnp.int32(0)
        self.scene = Scene()
        observation: Array = jnp.zeros((3, 3, 2), dtype=jnp.bool_)
        legal_action_mask: Array = _legal_actions(self.scene.turn_tracker, self.actions_remaining)

    @property
    def env_id(self) -> core.EnvId:
        return "5e_srd_dd_style"


def initiative(current_player, current_initiative, dex_ability_bonus, actions_remaining, bonus_actions_remaining):
    """
    the scheme for controlling the flow of initiative is

    1. assign each character an initiative score based on

        initiative = dex proficiency bonus o

        characters may have the same initiative number

        higher initiative goes first

        if tied: player 0 (the PC's) always go before player 1 (the NPC's)

        as the action space is duplicated for each player, we set actions in the current initiative number to legal,
        if player 0 and player 1 both have characters with same initiative, player 0 moves first, then player 1

    3.  player0 continues to choose actions and step the environment until all action resources are exhausted
        at the current initiative number, then player1 can choose actions for the NPCs at the same number

    4.  Once all actions are exhausted, play moves to the next valid initiative number

    5.  When the turn is over, the initiative is reset to the highest number

    :param dex_ability_bonus, current_initiative, actions_remaining, bonus_actions_remaining
    :return: legal_actions
    """

class TicTacToe(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

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


def _init(rng: PRNGKey) -> State:

    return State(current_player=current_player)  # type:ignore


def _step(state: State, action: Array) -> State:
    state = state.replace(_board=state._board.at[action].set(state._turn))  # type: ignore
    won = _win_check(state._board, state._turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    return state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        legal_action_mask=state._board < 0,
        rewards=reward,
        terminated=won | jnp.all(state._board != -1),
        _turn=(state._turn + 1) % 2,
    )


def _win_check(board, turn) -> Array:
    idx = jnp.int32([[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]])  # type: ignore
    return ((board[idx] == turn).all(axis=1)).any()


def _observe(state: State, player_id: Array) -> Array:
    @jax.vmap
    def plane(i):
        return (state._board == i).reshape((3, 3))

    # flip if player_id is opposite
    x = jax.lax.cond(
        state.current_player == player_id,
        lambda: jnp.int32([state._turn, 1 - state._turn]),
        lambda: jnp.int32([1 - state._turn, state._turn]),
    )

    return jnp.stack(plane(x), -1)