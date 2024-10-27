import constants
import dnd5e
from pgx.experimental import act_randomly
import jax
import jax.numpy as jnp
from constants import *
import turn_tracker
from constants import Abilities
import equipment.weapons as weapons


def test_sim():
    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    assert jnp.all(state.scene.party.ability_modifier[:, :, Abilities.DEX] == jnp.array([
        [1, 1, 3, -1],
        [1, 1, 3, -1]
    ]))

    assert jnp.all(state.scene.party.armor_class == jnp.array([
        [18, 18, 14, 9],
        [18, 18, 14, 9]
    ]))

    assert jnp.all(state.scene.party.actions.legal_use_pos[0, 0, Actions.ATTACK_MELEE_WEAPON] == jnp.array([
        [0, 0, 1, 1]
    ]))
    assert jnp.all(state.scene.party.actions.legal_target_pos[0, 0, Actions.ATTACK_MELEE_WEAPON] == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1]
    ]))
    assert state.scene.party.actions.damage[0, 0, Actions.ATTACK_MELEE_WEAPON] == 4.5
    assert state.scene.party.actions.damage_type[0, 0, Actions.ATTACK_MELEE_WEAPON] == constants.DamageType.SLASHING


    assert jnp.all(state.scene.party.actions.legal_use_pos[0, 0, Actions.ATTACK_RANGED_WEAPON] == jnp.array([1, 1, 0, 0]))
    assert jnp.all(state.scene.party.actions.legal_target_pos[0, 0, Actions.ATTACK_RANGED_WEAPON] == jnp.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ]))
    assert state.scene.party.actions.damage[0, 0, Actions.ATTACK_RANGED_WEAPON] == 4.5
    assert state.scene.party.actions.damage_type[0, 0, Actions.ATTACK_RANGED_WEAPON] == constants.DamageType.PIERCING


    action = dnd5e.encode_action(Actions.END_TURN, 0, 2, 0, 0)
    state = env.step(state, action)

    assert state.terminated == False
    assert state.scene.turn_tracker.initiative == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]))

    action = dnd5e.encode_action(Actions.END_TURN, 1, 2, 0, 0)
    state = env.step(state, action)

    assert state.terminated == False
    assert state.scene.turn_tracker.initiative == 1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [1, 1, 0, 0],
        [0, 0, 0, 0]
    ]))


def test_legal_actions_by_player_position():
    party = dnd5e.init_party()


def test_vmap():

    env = dnd5e.DND5E()

    # check vmap
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    rng_init_batch = jax.random.split(rng_init, 2)
    state = init(rng_init_batch)

    assert jnp.all(state.scene.party.ability_modifier[1, :, :, Abilities.DEX] == jnp.array([
        [1, 1, 3, -1],
        [1, 1, 3, -1]
    ]))

    action = dnd5e.encode_action(Actions.END_TURN, 0, 2, 0, 0)
    action = jnp.array([action, action])
    state = step(state, action)

    assert state.scene.turn_tracker.initiative[0] == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting[1] == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 0]
    ]))


    # while ~jnp.all(state.terminated):
    #     state = env.step(state, act_randomly(rng, state.legal_action_mask))
    #     jax.debug.print('{}', state._board)


def test_action_encode():
    action = jnp.array([Actions.END_TURN, Actions.ATTACK_MELEE_WEAPON])
    source_party = jnp.array([0, 1])
    source_character = jnp.array([2, 3])
    target_party = jnp.array([1, 0])
    target_slot = jnp.array([1, 2])
    enc_action = dnd5e.encode_action(action, source_party, source_character, target_party, target_slot)
    dsource_party, dsource_character, daction, dtarget_party, dtarget_slot = dnd5e.decode_action(enc_action)
    assert jnp.all(action == daction)
    assert jnp.all(source_party == dsource_party)
    assert jnp.all(source_character == dsource_character)
    assert jnp.all(target_party == dtarget_party)
    assert jnp.all(target_slot == dtarget_slot)

def test_legal_actions():

    scene = dnd5e.init_scene(None)
    legal_action_mask = dnd5e._legal_actions(scene)
    legal_action_mask = legal_action_mask.ravel()

    for character in range(N_CHARACTERS):
        action = dnd5e.encode_action(Actions.END_TURN, Party.PC, character, 0, 0)
        print(action, legal_action_mask[action])

    print(legal_action_mask)




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
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=TRUE)
    assert tt.initiative == 3
    assert tt.cohort[tt.party] == 0
    assert tt.turn[tt.party] == 0
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.initiative == 0
    assert tt.cohort[tt.party] == 2
    assert tt.turn[tt.party] == 0
    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 3
    assert tt.initiative == -1
    assert tt.turn[tt.party] == 0

    tt = turn_tracker.end_on_character_start(tt)

    tt = turn_tracker._next_cohort(tt, actions_remain=FALSE)
    assert tt.cohort[tt.party] == 0
    assert tt.initiative == 3
    assert tt.turn[tt.party] == 1

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
        assert jnp.all(tt.characters_acting == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # first end_turn action
        tt = turn_tracker.next_turn(tt, 0, 0)
        assert tt.party == 0
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [1, 0, 0, 1],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # second end turn action
        tt = turn_tracker.next_turn(tt, 0, 3)
        assert tt.party == 1
        assert tt.initiative == 3
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # third end turn action
        tt = turn_tracker.next_turn(tt, 1, 0)
        assert tt.party == 1
        assert tt.initiative == 1
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # fourth end turn action
        tt = turn_tracker.next_turn(tt, 1, 3)
        assert tt.party == 0
        assert tt.initiative == 0
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # fifth end turn action
        tt = turn_tracker.next_turn(tt, 0, 1)
        assert tt.party == 1
        assert tt.initiative == 0
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 0, 1, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # sixth end turn action
        tt = turn_tracker.next_turn(tt, 1, 2)
        assert tt.party == 0
        assert tt.initiative == -1
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 1, 0],
            [0, 0, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # seventh end turn action
        tt = turn_tracker.next_turn(tt, 0, 2)
        assert tt.party == 1
        assert tt.initiative == -1, f'{tt.initiative}'
        assert jnp.all(tt.characters_acting == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=jnp.bool))
        assert jnp.all(tt.on_character_start == jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0]
        ]))
        tt = turn_tracker.end_on_character_start(tt)

        # final end turn
        tt = turn_tracker.next_turn(tt, 1, 1)
