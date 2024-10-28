import chex
import jax.numpy as jnp
from jax import Array
from constants import N_PLAYERS, N_CHARACTERS

"""
the scheme for controlling the flow of initiative is

1. assign each character an initiative score based on

    initiative = dex proficiency bonus

    characters may have the same initiative number

    higher initiative goes first

    if tied: player 0 (the PC's) always go before player 1 (the NPC's)

2.  on_character_start is set when a character gains tha ability to act,

3.  PCs continue to choose actions and step the environment until all action resources are exhausted
    at the current initiative number, then NPCs can choose actions for the NPCs at the same number

4.  Once all actions are exhausted, play moves to the next valid initiative number

5.  When the turn is over, the initiative is reset to the highest number

"""


@chex.dataclass
class TurnTracker:
    initiative_scores: chex.ArrayDevice  # higher goes first
    turn_order: chex.ArrayDevice # index array that sorts parties by initiative
    end_turn: chex.ArrayDevice # if the end_turn button has been pressed
    party: chex.ArrayDevice # current party
    cohort: chex.ArrayDevice # which cohort in each party is current
    turn: chex.ArrayDevice # which turn each party is on
    on_character_start: chex.ArrayDevice  # trigger for events that occur on start of character turn

    @property
    def initiative(self):
        leading_dim = self.initiative_scores.shape[:-2]
        trailing_dim = self.initiative_scores.shape[-2:]
        leading_range = [jnp.arange(n) for n in leading_dim]
        party = self.party[*leading_range].squeeze()
        cohort = self.cohort[*leading_range, party].squeeze()
        turn_order = self.turn_order[*leading_range, party, cohort].squeeze()
        return self.initiative_scores[*leading_range, party, turn_order]

    @property
    def characters_acting(self):
        leading_dim = self.initiative_scores.shape[:-2]
        leading_range = [jnp.arange(n) for n in leading_dim]
        turn_mask = self.initiative_scores == self.initiative.reshape(leading_dim + (1, 1))
        turn_mask = turn_mask.at[*leading_range, (self.party + 1) % N_PLAYERS].set(False)  # set non active party to false
        return turn_mask


def init(dex_ability_bonus):
    initiative_scores = dex_ability_bonus
    on_character_start = jnp.zeros_like(initiative_scores, dtype=jnp.bool)
    on_character_start = on_character_start.at[0].set(initiative_scores[0].max() == initiative_scores[0])

    return TurnTracker(
        initiative_scores=initiative_scores,
        turn_order=jnp.argsort(initiative_scores, axis=-1, descending=True),
        end_turn=jnp.zeros_like(initiative_scores, dtype=jnp.bool),
        party=jnp.zeros(1, dtype=jnp.int32),
        cohort=jnp.zeros(N_PLAYERS, dtype=jnp.int32),
        turn=jnp.zeros(N_PLAYERS, dtype=jnp.int32),
        on_character_start=on_character_start
    )


def _next_cohort(turn_tracker, actions_remain: Array):
    # if more than 1 character has the same initiative, we must skip over them
    n_simultaneous_characters = jnp.sum(turn_tracker.initiative_scores[turn_tracker.party] == turn_tracker.initiative)
    next_cohort = (turn_tracker.cohort[turn_tracker.party] + n_simultaneous_characters) % N_CHARACTERS
    turn = jnp.where(next_cohort == 0, turn_tracker.turn[turn_tracker.party] + 1, turn_tracker.turn[turn_tracker.party])

    # only advance the current party
    next_cohort = turn_tracker.cohort.at[turn_tracker.party].set(next_cohort)
    next_turn = turn_tracker.turn.at[turn_tracker.party].set(turn)

    # advance only if all characters have ended turn
    turn_tracker.cohort = jnp.where(actions_remain, turn_tracker.cohort, next_cohort)
    turn_tracker.turn = jnp.where(actions_remain, turn_tracker.turn, next_turn)

    return turn_tracker


def next_turn(turn_tracker, end_turn, end_turn_party, end_turn_character):

    # if all characters ended turn, reset the counter
    turn_tracker.end_turn = jnp.where(jnp.all(turn_tracker.end_turn), jnp.zeros_like(turn_tracker.end_turn), turn_tracker.end_turn)
    turn_tracker.end_turn = turn_tracker.end_turn.at[end_turn_party, end_turn_character].set(jnp.bool(end_turn))

    # have any characters in the current round not ended their turn?
    actions_remain = jnp.any(turn_tracker.characters_acting & ~turn_tracker.end_turn)

    # advance the current party
    turn_tracker = _next_cohort(turn_tracker, actions_remain)

    # next party is the one with highest initiative or lowest turn
    R_PLAYERS = jnp.arange(N_PLAYERS)
    party_init = turn_tracker.initiative_scores[R_PLAYERS, turn_tracker.turn_order[R_PLAYERS, turn_tracker.cohort]]
    party_order = jnp.argmax(party_init)
    turn_order = jnp.argmin(turn_tracker.turn)
    next_party = jnp.where(turn_tracker.turn[0] == turn_tracker.turn[1], party_order, turn_order)

    # update only if the end_turn button was pressed for all characters
    turn_tracker.party = jnp.where(actions_remain, turn_tracker.party, next_party)

    # set the on_character_start for characters that just became active
    turn_tracker.on_character_start = jnp.where(actions_remain, turn_tracker.on_character_start, turn_tracker.characters_acting)

    return turn_tracker


def end_on_character_start(turn_tracker):
    """
    This should be called to clear the on_character_start flags after all events
    triggered on the start of a characters turn have been processed
    :param turn_tracker:
    :return: turn_tracker
    """
    turn_tracker.on_character_start = jnp.zeros_like(turn_tracker.on_character_start)
    return turn_tracker