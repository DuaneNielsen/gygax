import dnd5e
from pgx.experimental import act_randomly
import jax
import jax.numpy as jnp
from constants import ACTION_RESOURCE_TABLE, N_PLAYERS, N_ACTIONS, MAX_PARTY_SIZE, Actions, FALSE, TRUE
import turn_tracker


def test_sim():
    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    while ~jnp.all(state.terminated):
        state = env.step(state, act_randomly(rng, state.legal_action_mask))
        jax.debug.print('{}', state._board)

def test_action_encode():
    action = jnp.array([Actions.END_TURN, Actions.ATTACK_MELEE_WEAPON])
    source_party = jnp.array([0, 1])
    source_character = jnp.array([2, 3])
    target_party = jnp.array([1, 0])
    target_slot = jnp.array([1, 2])
    enc_action = dnd5e.encode_action(action, source_party, source_character, target_party, target_slot)
    daction, dsource_party, dsource_character, dtarget_party, dtarget_slot = dnd5e.decode_action(enc_action)
    assert jnp.all(action == daction)
    assert jnp.all(source_party == dsource_party)
    assert jnp.all(source_character == dsource_character)
    assert jnp.all(target_party == dtarget_party)
    assert jnp.all(target_slot == dtarget_slot)


def test_action_resource_table():
    action_resources_party1 = jnp.array([
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ])

    action_resources_party2 = jnp.array([
        [1, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ])

    action_resources = jnp.stack([action_resources_party1, action_resources_party2])

    actions_available = legal_actions_by_action_resource(action_resources)

    assert actions_available.shape == (N_PLAYERS, MAX_PARTY_SIZE, N_ACTIONS)

    assert jnp.all(
        actions_available[:, :, Actions.END_TURN] == jnp.array([
            [1, 1, 1, 0], [1, 1, 1, 0]
        ]))



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
    assert jnp.all(tt.on_character_start == jnp.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ]))
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=TRUE)
    assert tt.initiative == 3
    assert tt.cohort[tt.party] == 0
    assert tt.turn[tt.party] == 0
    assert jnp.all(tt.on_character_start == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]))
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.initiative == 0
    assert tt.cohort[tt.party] == 2
    assert tt.turn[tt.party] == 0
    assert jnp.all(tt.on_character_start == jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]))
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 3
    assert tt.initiative == -1
    assert tt.turn[tt.party] == 0
    assert jnp.all(tt.on_character_start == jnp.array([
        [0, 0, 1, 0],
        [0, 0, 0, 0]
    ]))
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 0
    assert tt.initiative == 3
    assert tt.turn[tt.party] == 1
    assert jnp.all(tt.on_character_start == jnp.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ]))
    tt = turn_tracker.end_on_character_start(tt)


def test_next_turn():
    dex_ability_bonus = jnp.array([
        [3, 0, -1, 3],
        [3, -1, 0, 1]
    ])

    tt = turn_tracker.init(dex_ability_bonus)

    for _ in range(3):

        # opening round
        assert tt.party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_turn == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # first end_turn action
        character_end_turn = jnp.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_turn == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # second end turn action
        character_end_turn = jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 1
        assert tt.initiative == 3
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # third end turn action
        character_end_turn = jnp.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 1
        assert tt.initiative == 1
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # fourth end turn action
        character_end_turn = jnp.array([
            [1, 0, 0, 1],
            [1, 0, 0, 1]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 0
        assert tt.initiative == 0
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # fifth end turn action
        character_end_turn = jnp.array([
            [1, 1, 0, 1],
            [1, 0, 0, 1]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 1
        assert tt.initiative == 0
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # sixth end turn action
        character_end_turn = jnp.array([
            [1, 1, 0, 1],
            [1, 0, 1, 1]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 0
        assert tt.initiative == -1
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # seventh end turn action
        character_end_turn = jnp.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)
        assert tt.party == 1
        assert tt.initiative == -1, f'{tt.initiative}'
        assert jnp.all(tt.characters_turn == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # final end turn
        character_end_turn = jnp.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        tt = turn_tracker.next_turn(tt, character_end_turn=character_end_turn)

