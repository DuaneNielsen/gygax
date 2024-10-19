import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

MAX_PARTY_SIZE = 4
N_PLAYERS = 2
N_WEAPON_SLOTS = 2
N_ABILITIES = 6

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

from enum import IntEnum


class Actions(IntEnum):
    END_TURN = 0
    DASH = 1
    ATTACK_MELEE_WEAPON = 3
    ATTACK_RANGED_WEAPON = 4


class ActionResourceUsageType(IntEnum):
    END_TURN = 0
    ACTION = 1
    BONUS_ACTION = 2
    ATTACK = 3
    SPELL_SLOT_1 = 4

class ActionResourceType(IntEnum):
    ACTION = 0
    BONUS_ACTION = 1
    ATTACK = 2

N_ACTIONS = len(Actions)
N_ACTION_RESOURCE_TYPES = len(ActionResourceUsageType)

action_resource_table = {
    Actions.END_TURN: ActionResourceUsageType.END_TURN,
    Actions.DASH: ActionResourceUsageType.ACTION,
    Actions.ATTACK_MELEE_WEAPON: ActionResourceUsageType.ATTACK,
    Actions.ATTACK_RANGED_WEAPON: ActionResourceUsageType.ATTACK
}

ACTION_RESOURCE_TABLE = jnp.zeros((N_ACTIONS), dtype=jnp.bool_)
for action, action_resource in action_resource_table.items():
    ACTION_RESOURCE_TABLE.at[action].set(action_resource)


def legal_actions_by_action_resource(action_resources):
    legal_actions = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS), dtype=jnp.bool)

    any_actions_remaining = (action_resources > 0).any(-1)
    legal_actions = legal_actions.at[:, :, 0].set(any_actions_remaining)
    # action_available = ACTION_RESOURCE_TABLE & action_resources.reshape(1, 1, 4)
    return legal_actions


def _legal_actions(turn_tracker, action_resources):
    legal_actions = jnp.ones((N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS), dtype=jnp.bool)
    resources_available = legal_actions_by_action_resource(action_resources)

    legal_actions = legal_actions & turn_tracker.characters_turn.unsqueeze(-1) & resources_available
    return legal_actions


class TurnTracker:

    def __init__(self, dex_ability_bonus):
        self.initiative_scores = dex_ability_bonus
        self.turn_order = jnp.argsort(self.initiative_scores, axis=-1, descending=True)
        self.party: Array = jnp.zeros(1, dtype=jnp.int32)
        self.cohort: Array = jnp.zeros(2, dtype=jnp.int32)
        self.turn: Array = jnp.zeros(2, dtype=jnp.int32)

    @property
    def initiative(self):
        return self.initiative_scores[jnp.arange(N_PLAYERS), self.turn_order[jnp.arange(N_PLAYERS), self.cohort]][self.party]

    @property
    def characters_turn(self):
        turn_mask = self.initiative_scores == self.initiative
        return turn_mask.at[(self.party + 1) % N_PLAYERS].set(False)

    def _next_cohort(self):

        # if more than 1 character has the same initiative, we must skip over them
        n_simultaneous_characters = jnp.sum(self.initiative_scores[self.party] == self.initiative)
        next_cohort = (self.cohort[self.party] + n_simultaneous_characters) % MAX_PARTY_SIZE
        turn = jnp.where(next_cohort == 0, self.turn[self.party] + 1, self.turn[self.party])

        # only advance the current party
        self.cohort = self.cohort.at[self.party].set(next_cohort)
        self.turn = self.turn.at[self.party].set(turn)
        return self

    def next_turn(self):
        # advance the current party
        self._next_cohort()

        # next party is the one with highest initiative or lowest turn
        party_init = self.initiative_scores[jnp.arange(N_PLAYERS), self.turn_order[jnp.arange(N_PLAYERS), self.cohort]]
        party_order = jnp.argmax(party_init)
        turn_order = jnp.argmin(self.turn)
        self.party = jnp.where(self.turn[0] == self.turn[1], party_order, turn_order)

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

@dataclass
class Scene:
    player_pos: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)
    party_0: Party = Party()
    party_1: Party = Party()
    current_initiative: Array = jnp.int32(0)
    initiative_score: Array = jnp.zeros((N_PLAYERS, MAX_PARTY_SIZE), dtype=jnp.int32)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((3, 3, 2), dtype=jnp.bool_)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(9, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- Tic-tac-toe specific ---
    _turn: Array = jnp.int32(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    _board: Array = -jnp.ones(9, jnp.int32)  # -1 (empty), 0, 1

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
    current_player = jnp.int32(jax.random.bernoulli(rng))
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