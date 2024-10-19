import sim
from pgx.experimental import act_randomly
import jax
import jax.numpy as jnp
from sim import ACTION_RESOURCE_TABLE, legal_actions_by_action_resource, N_PLAYERS, N_ACTIONS, MAX_PARTY_SIZE, Actions
from sim import legal_actions_by_initiative, TurnTracker


def test_sim():
    env = sim.TicTacToe()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    while ~jnp.all(state.terminated):
        state = env.step(state, act_randomly(rng, state.legal_action_mask))
        jax.debug.print('{}', state._board)


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


def test_legal_initiative():
    dex_ability_bonus = jnp.array([3, 0, -1, 3])
    dex_ability_bonus = jnp.stack([dex_ability_bonus, dex_ability_bonus])

    current_initiative = jnp.array([3])
    current_player = jnp.array([0])

    characters_moving = legal_actions_by_initiative(dex_ability_bonus, current_initiative, current_player)

    assert characters_moving.shape == (2, 4)

    assert jnp.all(characters_moving == jnp.array([
        [1, 0, 0, 1],
        [0, 0, 0, 0]
    ], dtype=jnp.bool))

    current_player = jnp.array([1])
    characters_moving = legal_actions_by_initiative(dex_ability_bonus, current_initiative, current_player)

    assert jnp.all(characters_moving == jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 1]
    ], dtype=jnp.bool))

    current_initiative = jnp.array([-1])
    characters_moving = legal_actions_by_initiative(dex_ability_bonus, current_initiative, current_player)

    assert jnp.all(characters_moving == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ], dtype=jnp.bool))


def test_next_cohort():
    dex_ability_bonus = jnp.array([
        [3, 0, -1, 3],
        [3, 0, -1, 3]
    ])

    turn_tracker = TurnTracker(dex_ability_bonus)
    assert jnp.all(turn_tracker.cohort == jnp.zeros(2, dtype=jnp.int32))
    assert jnp.all(turn_tracker.turn_order == jnp.array([0, 3, 1, 2]))
    assert jnp.all(turn_tracker.turn == jnp.all(jnp.zeros(2, dtype=jnp.int32)))
    assert turn_tracker.initiative == 3

    turn_tracker = turn_tracker._next_cohort()
    assert turn_tracker.initiative == 0
    assert turn_tracker.cohort[turn_tracker.party] == 2
    assert turn_tracker.turn[turn_tracker.party] == 0

    turn_tracker = turn_tracker._next_cohort()
    assert turn_tracker.cohort[turn_tracker.party] == 3
    assert turn_tracker.initiative == -1
    assert turn_tracker.turn[turn_tracker.party] == 0

    turn_tracker = turn_tracker._next_cohort()
    assert turn_tracker.cohort[turn_tracker.party] == 0
    assert turn_tracker.initiative == 3
    assert turn_tracker.turn[turn_tracker.party] == 1


def test_next_turn():
    dex_ability_bonus = jnp.array([
        [3, 0, -1, 3],
        [3, -1, 0, 1]
    ])

    tt = TurnTracker(dex_ability_bonus)

    for _ in range(3):
        assert tt.party == 0
        assert tt.initiative == 3

        tt.next_turn()
        assert tt.party == 1
        assert tt.initiative == 3

        tt.next_turn()
        assert tt.party == 1
        assert tt.initiative == 1

        tt.next_turn()
        assert tt.party == 0
        assert tt.initiative == 0

        tt.next_turn()
        assert tt.party == 1
        assert tt.initiative == 0

        tt.next_turn()
        assert tt.party == 0
        assert tt.initiative == -1

        tt.next_turn()
        assert tt.party == 1
        assert tt.initiative == -1, f'{tt.initiative}'

        tt.next_turn()


