import jax.numpy as jnp
import pytest
import turn_tracker
from constants import TRUE, FALSE

def test_next_cohort():
    dex_ability_bonus = jnp.array([
        [3, 0, -1, 3],
        [3, 0, -1, 3]
    ])

    tt = turn_tracker.init(dex_ability_bonus)
    assert jnp.all(tt.cohort == jnp.zeros(2, dtype=jnp.int32))
    assert jnp.all(tt.turn_order == jnp.array([0, 3, 1, 2]))
    assert jnp.all(tt.turn == jnp.all(jnp.zeros(2, dtype=jnp.int32)))
    assert tt.initiative == 3
    tt = turn_tracker.clear_events(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=TRUE)
    assert tt.initiative == 3
    assert tt.cohort[tt.party] == 0
    assert tt.turn[tt.party] == 0
    tt = turn_tracker.clear_events(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.initiative == 0
    assert tt.cohort[tt.party] == 2
    assert tt.turn[tt.party] == 0
    tt = turn_tracker.clear_events(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 3
    assert tt.initiative == -1
    assert tt.turn[tt.party] == 0

    tt = turn_tracker.clear_events(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 0
    assert tt.initiative == 3
    assert tt.turn[tt.party] == 1

    tt = turn_tracker.clear_events(tt)


def test_next_turn():
    dex_ability_bonus = jnp.array([
        [3, 0, -1, 3],
        [3, -1, 0, 1]
    ])

    tt = turn_tracker.init(dex_ability_bonus)

    for _ in range(3):
        # opening round
        assert tt.on_turn_start == True
        assert tt.party == 0
        assert tt.prev_party == 1
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ]))
        assert jnp.all(tt.end_turn == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))

        tt = turn_tracker.clear_events(tt)
        assert tt.on_turn_start == False

        # first action, dont end turn
        tt = turn_tracker.next_turn(tt, False, 0, 0)
        assert tt.on_turn_start == False
        assert tt.party == 0
        assert tt.prev_party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # first end_turn action
        tt = turn_tracker.next_turn(tt, True, 0, 0)
        assert tt.party == 0
        assert tt.prev_party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))

        tt = turn_tracker.clear_events(tt)

        # second end turn action
        tt = turn_tracker.next_turn(tt, True, 0, 3)
        assert tt.party == 1
        assert tt.prev_party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # third end turn action
        tt = turn_tracker.next_turn(tt, True, 1, 0)
        assert tt.party == 1
        assert tt.prev_party == 1
        assert tt.initiative == 1
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]))
        tt = turn_tracker.clear_events(tt)

        # fourth end turn action
        tt = turn_tracker.next_turn(tt, False, 1, 3)
        assert tt.party == 1
        assert tt.prev_party == 1
        assert tt.initiative == 1
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        tt = turn_tracker.next_turn(tt, True, 1, 3)
        assert tt.party == 0
        assert tt.prev_party == 1
        assert tt.initiative == 0
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # fifth end turn action
        tt = turn_tracker.next_turn(tt, True, 0, 1)
        assert tt.party == 1
        assert tt.prev_party == 0
        assert tt.initiative == 0
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # sixth end turn action
        tt = turn_tracker.next_turn(tt, True, 1, 2)
        assert tt.party == 0
        assert tt.prev_party == 1
        assert tt.initiative == -1
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # seventh end turn action
        tt = turn_tracker.next_turn(tt, True, 0, 2)
        assert tt.party == 1
        assert tt.prev_party == 0
        assert tt.initiative == -1, f'{tt.initiative}'
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ]))
        tt = turn_tracker.clear_events(tt)

        # final end turn
        tt = turn_tracker.next_turn(tt, True, 1, 1)

