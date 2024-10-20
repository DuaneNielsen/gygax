import chex
import jax.numpy as jnp
from jax import Array
from constants import N_PLAYERS, MAX_PARTY_SIZE


@chex.dataclass
class TurnTracker:
    initiative_scores: chex.ArrayDevice
    turn_order: chex.ArrayDevice
    end_turn: chex.ArrayDevice
    party: chex.ArrayDevice
    cohort: chex.ArrayDevice
    turn: chex.ArrayDevice

    @property
    def initiative(self):
        return self.initiative_scores[jnp.arange(N_PLAYERS), self.turn_order[jnp.arange(N_PLAYERS), self.cohort]][
            self.party]

    @property
    def characters_turn(self):
        turn_mask = self.initiative_scores == self.initiative
        return turn_mask.at[(self.party + 1) % N_PLAYERS].set(False)


def init(dex_ability_bonus):
    return TurnTracker(
        initiative_scores=dex_ability_bonus,
        turn_order=jnp.argsort(dex_ability_bonus, axis=-1, descending=True),
        end_turn=jnp.zeros_like(dex_ability_bonus, dtype=jnp.bool_),
        party=jnp.zeros(1, dtype=jnp.int32),
        cohort=jnp.zeros(N_PLAYERS, dtype=jnp.int32),
        turn=jnp.zeros(N_PLAYERS, dtype=jnp.int32)
    )


def _next_cohort(turn_tracker, actions_remain: Array):
    # if more than 1 character has the same initiative, we must skip over them
    n_simultaneous_characters = jnp.sum(turn_tracker.initiative_scores[turn_tracker.party] == turn_tracker.initiative)
    next_cohort = (turn_tracker.cohort[turn_tracker.party] + n_simultaneous_characters) % MAX_PARTY_SIZE
    turn = jnp.where(next_cohort == 0, turn_tracker.turn[turn_tracker.party] + 1, turn_tracker.turn[turn_tracker.party])

    # only advance the current party
    next_cohort = turn_tracker.cohort.at[turn_tracker.party].set(next_cohort)
    next_turn = turn_tracker.turn.at[turn_tracker.party].set(turn)

    # advance only if all characters have ended turn
    turn_tracker.cohort = jnp.where(actions_remain, turn_tracker.cohort, next_cohort)
    turn_tracker.turn = jnp.where(actions_remain, turn_tracker.turn, next_turn)
    return turn_tracker


def next_turn(turn_tracker, character_end_turn):
    # have any characters not ended their turn?
    actions_remain = jnp.any(turn_tracker.characters_turn & ~character_end_turn)

    # advance the current party
    turn_tracker = _next_cohort(turn_tracker, actions_remain)

    # next party is the one with highest initiative or lowest turn
    party_init = turn_tracker.initiative_scores[
        jnp.arange(N_PLAYERS), turn_tracker.turn_order[jnp.arange(N_PLAYERS), turn_tracker.cohort]]
    party_order = jnp.argmax(party_init)
    turn_order = jnp.argmin(turn_tracker.turn)
    next_party = jnp.where(turn_tracker.turn[0] == turn_tracker.turn[1], party_order, turn_order)

    # update only if the end_turn button was pressed for all characters at the current initiative
    turn_tracker.party = jnp.where(actions_remain, turn_tracker.party, next_party)
    return turn_tracker
