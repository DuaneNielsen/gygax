import pytest

import actions
import constants
import dnd5e
from pgx.experimental import act_randomly
import jax

from conditions import Conditions
from constants import *
import turn_tracker
from constants import Abilities
import default_config


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


@pytest.fixture
def state():
    state = dnd5e._init(jax.random.PRNGKey(0), default_config.default_config)
    return state


"""
starting postions ['fizban', 'jimmy', 'goldmoon', 'riverwind'], ['raistlin', 'joffrey', 'clarion', 'pikachu']
"""

fizban, jimmy, goldmoon, riverwind = (0, 0), (0, 1), (0, 2), (0, 3)
raistlin, joffrey, clarion, pikachu = (1, 0), (1, 1), (1, 2), (1, 3)

characters = {
    'fizban': fizban,
    'jimmy': jimmy,
    'goldmoon': goldmoon,
    'riverwind': riverwind,
    'raistlin': raistlin,
    'joffrey': joffrey,
    'clarion': clarion,
    'pikachu': pikachu
}


def test_state_init(rng):
    state = dnd5e._init(rng, default_config.default_config)
    assert state.current_player == 0
    assert state.terminated == False
    assert state.rewards[0] == 0.
    assert state.rewards[1] == 0.
    assert state.names[0, 1] == 'jimmy'


def test_env_init():
    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    assert jnp.all(state.party.ability_modifier.dexterity[:, :] == jnp.array([
        [-1, 3, 1, 1],
        [-1, 3, 1, 1]
    ]))

    assert jnp.all(state.party.armor_class == jnp.array([
        [9, 14, 18, 18],
        [9, 14, 18, 18]
    ]))

    assert state.party.main_attack.expected_damage[0, 3] == 7.5
    assert state.party.main_attack.damage_type[0, 3] == constants.DamageType.SLASHING

    assert state.party.ranged_main_attack.expected_damage[0, 3] == 4.5
    assert state.party.ranged_main_attack.damage_type[0, 3] == constants.DamageType.PIERCING


def test_env_step():
    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    # first move, jimmy attacks pikachu
    assert state.current_player == 0
    assert state.turn_tracker.initiative == 3
    assert jnp.all(state.turn_tracker.characters_acting == jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]))
    assert state.action_resources.current.action[*jimmy] == 1

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    assert state.legal_action_mask[action] == True

    state = env.step(state, action)
    assert state.terminated == False
    assert state.scene.party.hitpoints[1, 3] == 13 - 3.5
    assert state.scene.party.action_resources[*jimmy, ActionResourceType.ACTION] == 0

    # end turn after taking first move

    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*jimmy, ActionResourceType.ACTION] == 0
    assert state.terminated == False

    # second move joffrey attacks jimmy
    assert state.current_player == 1
    assert state.scene.turn_tracker.initiative == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]))

    assert jnp.all(
        state.scene.party.actions.legal_use_pos[1, 2, Actions.ATTACK_RANGED_WEAPON] == jnp.array([1, 1, 0, 0]))
    assert jnp.all(state.scene.party.actions.legal_target_pos[1, 2, Actions.ATTACK_RANGED_WEAPON] == jnp.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1]
    ]))
    assert state.scene.party.action_resources[*joffrey, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 1)
    assert state.legal_action_mask[action] == True

    state = env.step(state, action)
    assert state.terminated == False
    assert state.scene.party.action_resources[*joffrey, ActionResourceType.ACTION] == 0
    assert state.scene.turn_tracker.initiative == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]))
    assert state.scene.party.hitpoints[*jimmy] == 8.0 - 3.5

    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)
    assert state.terminated == False

    # third move riverwind attacks pikachu
    assert state.current_player == 0
    assert state.scene.turn_tracker.initiative == 1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ]))
    assert state.scene.party.action_resources[*riverwind, ActionResourceType.ACTION] == 1
    legal_action_mask = state.legal_action_mask.reshape(N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS)
    assert legal_action_mask[riverwind[1], Actions.END_TURN, 0, 0] == True
    assert legal_action_mask[goldmoon[1], Actions.END_TURN, 0, 0] == True

    action = dnd5e.encode_action(Actions.ATTACK_MELEE_WEAPON, 3, TargetParty.ENEMY, 3)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*riverwind, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[*pikachu] == 13 - 3.5 - 4.5
    action = dnd5e.encode_action(Actions.END_TURN, 3, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # fourth move - goldmoon attacks pikachu

    assert state.current_player == 0
    assert state.scene.turn_tracker.initiative == 1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 1, 1],
        [0, 0, 0, 0]
    ]))
    legal_action_mask = state.legal_action_mask.reshape(N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS)
    assert jnp.all(legal_action_mask[riverwind[1], Actions.END_TURN, 0, 0]) == False
    assert jnp.all(legal_action_mask[goldmoon[1], Actions.END_TURN, 0, 0]) == True

    assert state.scene.party.action_resources[*goldmoon, ActionResourceType.ACTION] == 1
    action = dnd5e.encode_action(Actions.ATTACK_MELEE_WEAPON, 2, TargetParty.ENEMY, 3)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*goldmoon, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[*pikachu] == 13 - 3.5 - 4.5 - 3.5
    action = dnd5e.encode_action(Actions.END_TURN, 2, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # fifth move pikachu attacks goldmoon
    assert state.current_player == 1
    assert state.scene.turn_tracker.initiative == 1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1]
    ]))
    assert state.scene.party.action_resources[*pikachu, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_MELEE_WEAPON, 3, TargetParty.ENEMY, 2)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*pikachu, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[0, 2] == 11 - 4.5

    action = dnd5e.encode_action(Actions.END_TURN, 3, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # sixth move - clarion attacks goldmoon
    assert state.current_player == 1
    assert state.scene.turn_tracker.initiative == 1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 0, 1, 1]
    ]))
    assert state.scene.party.action_resources[*clarion, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_MELEE_WEAPON, 2, TargetParty.ENEMY, 2)
    assert state.legal_action_mask[action] == True

    state = env.step(state, action)
    assert state.scene.party.action_resources[*clarion, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[0, 2] == 11 - 4.5 - 3.5

    action = dnd5e.encode_action(Actions.END_TURN, 2, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # seventh move - fizban shoots pikachu
    assert state.current_player == 0
    assert state.scene.turn_tracker.initiative == -1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0]
    ]))
    assert state.scene.party.action_resources[*fizban, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 0, TargetParty.ENEMY, 3)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*fizban, ActionResourceType.ACTION] == 0

    assert state.scene.party.hitpoints[*pikachu] == 13 - 3.5 - 4.5 - 3.5 - 3.5
    assert state.scene.party.conditions[*pikachu, Conditions.DEAD] == True
    assert state.scene.party.action_resources[*pikachu].sum() == 0
    action = dnd5e.encode_action(Actions.END_TURN, 0, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # eighth move - raistlin shoots goldmoon
    assert state.current_player == 1
    assert state.scene.turn_tracker.initiative == -1
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0]
    ]))
    assert state.scene.party.action_resources[*raistlin, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 0, TargetParty.ENEMY, 2)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*raistlin, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[0, 2] == 11 - 4.5 - 3.5 - 3.5
    assert state.scene.party.conditions[*goldmoon, Conditions.DEAD] == True
    assert state.scene.party.action_resources[*goldmoon].sum() == 0

    action = dnd5e.encode_action(Actions.END_TURN, 0, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # ninth move, jimmy shoots raistlin
    print('ninth move, jimmy shoots raistlin')
    assert state.scene.turn_tracker.on_turn_start == False
    assert state.current_player == 0
    assert state.scene.turn_tracker.initiative == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]))
    assert state.scene.party.action_resources[*jimmy, ActionResourceType.ACTION] == 1

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 0)
    assert state.legal_action_mask[action] == True
    state = env.step(state, action)
    assert state.scene.party.action_resources[*jimmy, ActionResourceType.ACTION] == 0
    assert state.scene.party.hitpoints[1, 0] == 6 - 3.5
    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    # tenth move
    assert state.current_player == 1
    assert state.scene.turn_tracker.initiative == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting == jnp.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]))

    for name, index in characters.items():
        print(name, actions.item())


def test_legal_actions():
    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    state = env.step(state, action)
    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    state = env.step(state, action)
    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)
    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 1)
    state = env.step(state, action)
    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    state = env.step(state, action)

    legal_actions = dnd5e._legal_actions(state)

    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 0] == False
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 1] == False
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 2] == True
    assert legal_actions[0, 2, Actions.ATTACK_MELEE_WEAPON, 1, 3] == True

    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 0, 0] == False
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 1] == False
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 2] == False
    assert legal_actions[0, 2, Actions.ATTACK_RANGED_WEAPON, 1, 3] == False

    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 0, 0] == False
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 1] == False
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 2] == False
    assert legal_actions[0, 0, Actions.ATTACK_MELEE_WEAPON, 1, 3] == False

    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 0, 0] == False
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 1] == False
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 2] == False
    assert legal_actions[0, 0, Actions.ATTACK_RANGED_WEAPON, 1, 3] == False


def test_vmap():
    env = dnd5e.DND5E()

    # check vmap
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    rng_init_batch = jax.random.split(rng_init, 2)
    state = init(rng_init_batch)

    assert jnp.all(state.scene.party.ability_modifier[1, :, :, Abilities.DEX] == jnp.array([
        [-1, 3, 1, 1],
        [-1, 3, 1, 1]
    ]))

    # jimmy shoots pikachu

    assert state.current_player[0] == 0
    assert state.scene.turn_tracker.initiative[0] == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting[0] == jnp.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0]
    ]))

    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    assert state.legal_action_mask[0, action] == True
    state = step(state, jnp.array([action, action]))
    assert state.scene.party.hitpoints[0, 1, 3] == 13 - 3.5

    action = dnd5e.encode_action(Actions.END_TURN, 1, TargetParty.FRIENDLY, 0)
    state = step(state, jnp.array([action, action]))

    # joffrey shoots

    assert state.scene.turn_tracker.initiative[0] == 3
    assert jnp.all(state.scene.turn_tracker.characters_acting[1] == jnp.array([
        [0, 0, 0, 0],
        [0, 1, 0, 0]
    ]))

    i = 0

    while ~jnp.all(state.terminated):
        rng, rng_policy = jax.random.split(rng)
        state = step(state, act_randomly(rng_policy, state.legal_action_mask))
        jax.debug.print('{}', state.terminated)
        i += 1
        if i > 10:
            break


def test_action_encode():
    action = jnp.array([Actions.END_TURN, Actions.ATTACK_MELEE_WEAPON])
    current_party = jnp.array([Party.PC, Party.NPC])
    source_character = jnp.array([2, 3])
    target_party = jnp.array([TargetParty.ENEMY, TargetParty.FRIENDLY])
    target_slot = jnp.array([1, 2])
    pos = jnp.array([
        [0, 1, 2, 3],
        [3, 2, 1, 0]
    ])
    enc_action = dnd5e.encode_action(action, source_character, target_party, target_slot)
    dec_action = dnd5e.decode_action(enc_action, current_party, pos)
    assert jnp.all(action == dec_action.action)
    assert jnp.all(current_party == dec_action.source.party)
    assert jnp.all(source_character == dec_action.source.index)
    assert jnp.all((target_party + current_party) % 2 == dec_action.target_slot.party)  # the target party switches depending upon the player
    assert jnp.all(target_slot == dec_action.target_slot.slot)
    assert dec_action.target.index[0] == 2
    assert dec_action.target.index[1] == 1



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


def test_apply_death(state):
    state.scene.party.hitpoints = state.scene.party.hitpoints.at[fizban].set(0)
    state = dnd5e.apply_death(state)
    assert state.scene.party.conditions[*fizban, Conditions.DEAD] == 1
    assert state.scene.party.action_resources[*fizban].sum() == 0


def test_win_check(state):
    state.scene.party.conditions = state.scene.party.conditions.at[*fizban, Conditions.DEAD].set(1)
    game_over, winner = dnd5e._win_check(state)
    assert game_over == False

    state.scene.party.conditions = state.scene.party.conditions.at[*raistlin, Conditions.DEAD].set(1)
    assert game_over == False

    state.scene.party.conditions = state.scene.party.conditions.at[*riverwind, Conditions.DEAD].set(1)
    state.scene.party.conditions = state.scene.party.conditions.at[*jimmy, Conditions.DEAD].set(1)
    state.scene.party.conditions = state.scene.party.conditions.at[*clarion, Conditions.DEAD].set(1)
    state.scene.party.conditions = state.scene.party.conditions.at[*goldmoon, Conditions.DEAD].set(1)

    game_over, winner = dnd5e._win_check(state)
    assert game_over == True
    assert winner == 1

    state.scene.party.conditions = state.scene.party.conditions.at[*riverwind, Conditions.DEAD].set(0)
    state.scene.party.conditions = state.scene.party.conditions.at[*raistlin, Conditions.DEAD].set(1)
    state.scene.party.conditions = state.scene.party.conditions.at[*pikachu, Conditions.DEAD].set(1)
    state.scene.party.conditions = state.scene.party.conditions.at[*joffrey, Conditions.DEAD].set(1)

    game_over, winner = dnd5e._win_check(state)
    assert game_over == True
    assert winner == 0


def test_reward_wrapper():
    env = dnd5e.DND5E()
    env = dnd5e.wrap_reward_on_hitbar_percentage(env)
    state = env.init(jax.random.PRNGKey(0))
    action = dnd5e.encode_action(Actions.ATTACK_RANGED_WEAPON, 1, TargetParty.ENEMY, 3)
    state = env.step(state, action)
    assert jnp.all(jnp.abs(state.rewards - jnp.array([0.0673077, 0.], dtype=jnp.float32)) < 0.000001)


def test_wrap_party_initiative():
    env = dnd5e.DND5E()
    state = env.init(jax.random.PRNGKey(0))
    wrapped_env = dnd5e.wrap_party_initiative(env, 0, -1)
    wrapped_state = wrapped_env.init(jax.random.PRNGKey(0))

    assert jnp.allclose(state.scene.turn_tracker.initiative_scores[0] - 1, wrapped_state.scene.turn_tracker.initiative_scores[0])
    assert jnp.allclose(state.scene.turn_tracker.initiative_scores[1], wrapped_state.scene.turn_tracker.initiative_scores[1])
