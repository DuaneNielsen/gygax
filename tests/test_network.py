import jax
from flax import nnx
import jax.numpy as jnp

import train
from constants import *
from funcs import create_argmin_mask
import pickle
import dnd5e
from pgx.experimental import auto_reset
from pathlib import Path
from random import shuffle
from train import train_step, MLP
import numpy as onp
import optax
from tree_serialization import flatten_pytree_batched
import pytest
import random

vmap_flatten = jax.vmap(flatten_pytree_batched)


class Linear(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear = nnx.Linear(din, dout, rngs=rngs, use_bias=False)

    def __call__(self, x):
        return self.linear(x)


def target_value_signal(state):
    N, P, C, S = state.observation.party.hitpoints.shape
    return jnp.sum(state.observation.party.hitpoints.reshape(N, P * C * S), -1, keepdims=True)



def target_policy_signal(state, rng):
    jnp.set_printoptions(threshold=float('inf'), linewidth=150, precision=4)
    legal_actions = state.legal_action_mask.reshape(N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS)

    legal_melee_attack = legal_actions[:, Actions.ATTACK_MELEE_WEAPON]
    legal_ranged_attack = legal_actions[:, Actions.ATTACK_RANGED_WEAPON]
    hitpoints = state.observation.party.hitpoints.sum(-1)[None, ...]

    # jax.debug.print('melee attack {}', legal_ranged_attack)
    # jax.debug.print('ranged attack {}', legal_ranged_attack)
    # jax.debug.print('hitpoints {}', hitpoints)

    valid_target = (hitpoints > 0) & (legal_melee_attack | legal_ranged_attack)

    end_turn_policy = jnp.zeros_like(legal_actions, dtype=jnp.float32)
    end_turn_policy = end_turn_policy.at[:, Actions.END_TURN].set(1.)

    valid_target_health = jnp.where(valid_target, hitpoints, 20)
    attacK_target_policy = create_argmin_mask(valid_target_health)
    attacK_target_policy = attacK_target_policy[:, None, ...]
    attacK_target_policy = attacK_target_policy * legal_actions

    has_valid_target = attacK_target_policy.sum() > 0

    # jax.debug.print('has_valid_target {}', has_valid_target)

    policy = jnp.where(has_valid_target, attacK_target_policy, end_turn_policy)

    # jax.debug.print('{}', policy.sum())
    policy = policy * legal_actions
    policy /= policy.sum()
    policy = policy.ravel()
    action = jnp.argmax(policy, -1)
    jax.debug.print('{}', action)
    policy = jax.nn.one_hot(action, policy.shape[-1])
    # jax.debug.print('{}', policy)
    # jax.debug.print('{}', policy.shape)
    return policy


@pytest.fixture
def env():
    return dnd5e.DND5E()

@pytest.fixture
def model(env):
    rng_env, rng_model = jax.random.split(jax.random.PRNGKey(0))
    state = env.init(rng_env)
    observation_features = flatten_pytree_batched(state.observation).shape[0]
    model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    return model

def test_policy_signal():
    env = dnd5e.DND5E()
    rng_env, rng_target = jax.random.split(jax.random.PRNGKey(0))
    state = env.init(rng_env)
    policy = target_policy_signal(state, rng_target)
    assert policy.shape[0] == N_CHARACTERS * N_ACTIONS * N_PLAYERS * N_CHARACTERS


def generate_data(rng_key, batch_size, dataset_size):

    vmap_target_policy_signal = jax.vmap(target_policy_signal)
    rng_key, rng_policy, rng_env = jax.random.split(rng_key, 3)
    from matplotlib import pyplot as plt
    from matplotlib import gridspec
    from play import CharacterDataVisualizer
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1.5])
    fig.suptitle(f'D&D 5e {"party"} State', size=16)

    # generate data
    buffer = []

    env = dnd5e.DND5E()
    state = jax.vmap(env.init)(jax.random.split(rng_env, batch_size))

    observation = vmap_flatten(state.observation)
    target_value = target_value_signal(state)
    target_policy = vmap_target_policy_signal(state, jax.random.split(rng_policy, batch_size))
    legal_actions = state.legal_action_mask
    current_player = state.current_player

    # Create character data visualization
    character_visualizer = CharacterDataVisualizer(fig, gs, jax.tree.map(lambda s: s[0], state))
    character_visualizer.create_visualization()
    plt.pause(0.01)

    for o, p, v, l, c in zip(observation, target_policy, target_value, legal_actions, current_player):
        buffer.append((o, p, v, l, c))

    for i in range(dataset_size - 1):
        rng_key, rng_policy, rng_env = jax.random.split(rng_key, 3)
        keys = jax.random.split(rng_env, batch_size)
        action = jnp.argmax(target_policy, -1)
        state = jax.vmap(auto_reset(env.step, env.init))(state, action, keys)
        jax.debug.print('terminated {}', state.terminated)
        character_visualizer.refresh(jax.tree.map(lambda s: s[0], state))
        plt.pause(0.02)

        observation = vmap_flatten(state.observation)
        target_value = target_value_signal(state)
        target_policy = vmap_target_policy_signal(state, jax.random.split(rng_policy, batch_size))
        legal_actions = state.legal_action_mask
        current_player = state.current_player

        for o, p, v, l, c in zip(observation, target_policy, target_value, legal_actions, current_player):
            buffer.append((o, p, v, l, c))
    return buffer


def generate_synthetic_data():
    from tree_serialization import cum_bins
    target = jnp.concatenate([jnp.arange(20) for _ in range(5)])
    observation = cum_bins(target, 19)
    target = target[:, None]
    buffer = []
    for o, t in zip(observation, target):
        buffer.append((o, t))
    return buffer


def test_network():
    batch_size = 16
    dataset_size = 100
    num_epochs = 2**8
    learning_rate = 1e-3
    hidden_dim = 2**7

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model = jax.random.split(rng_key, 2)

    # generate the dataset
    dataset_file = Path('buffer.pkl')

    if dataset_file.exists():
        with dataset_file.open('rb') as f:
            buffer = pickle.load(f)
    else:
        buffer = generate_data(rng_key, batch_size, dataset_size)
        with dataset_file.open('wb') as f:
            pickle.dump(buffer, f)

    random.seed(0)
    shuffle(buffer)
    observation, policy, value, legal_actions, current_player = tuple([jnp.stack(d) for d in zip(*buffer)])
    max_target = jnp.max(value)
    target = value / max_target

    # train test split
    test_split = len(buffer) // 10
    train_observation, test_observation = observation[:-test_split], observation[-test_split:]
    train_policy, test_policy = policy[:-test_split], policy[-test_split:]
    train_value, test_value = value[:-test_split], value[-test_split:]
    train_legal, test_legal = legal_actions[:-test_split], legal_actions[-test_split:]
    train_current_player, test_current_player = current_player[:-test_split], current_player[-test_split:]
    assert jnp.all(test_policy.sum(-1) == 1.)

    # split into minibatches
    def batchify(d):
        return jnp.stack(jnp.split(d, d.shape[0] // batch_size))

    train_observation, train_policy, train_value = batchify(train_observation), batchify(train_policy), batchify(train_value)

    env = dnd5e.DND5E()
    num_actions = env.num_actions

    model = MLP(observation.shape[1], hidden_dim, num_actions, rngs=nnx.Rngs(rng_model))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    policy, value = model(train_observation[0])
    value = value.squeeze(-1)
    assert value.shape == train_value[0].shape
    assert policy.shape == train_policy[0].shape

    # Create the plot
    # plot = LiveProbabilityPlot(num_probabilities=model.linear.kernel.value.shape[0])
    # plot.update(model.linear.kernel.value[:, 0])
    # jax.debug.print('{}', dense_0)
    # time.sleep(0.1)

    from statistics import mean

    for epoch in range(num_epochs):
        losses = []
        for obs, target_policy, target_value in zip(train_observation, train_policy, train_value):
            loss = train_step(model, optimizer, obs, target_policy, target_value)
            losses.append(loss.item())
            # plot.update(model.linear.kernel.value[:, 0])
        print(f'loss {mean(losses)}')

    from math import prod

    policy, value = model(test_observation)
    value = value.squeeze(-1)
    policy = policy * test_legal
    test_loss = jnp.mean((value - test_value) ** 2)
    jax.debug.print('{}', value.squeeze())
    jax.debug.print('{}', test_value.squeeze())

    jax.debug.print('{}', jnp.argmax(policy.squeeze(), -1))
    jax.debug.print('{}', jnp.argmax(test_policy.squeeze(), -1))

    assert jnp.all(test_policy.sum(-1) == 1.)

    policy = onp.array(policy.squeeze())
    test_policy = onp.array(test_policy)
    assert onp.all(test_policy.sum(-1) == 1.)
    test_current_player = onp.array(test_current_player)
    onp.savetxt("../analysis/policy.csv", policy, delimiter=",")
    onp.savetxt("../analysis/test_policy.csv", test_policy, delimiter=",")
    onp.savetxt("../analysis/test_current_player.csv", test_current_player, delimiter=",")

    policy_loss = jnp.mean(optax.softmax_cross_entropy(policy, test_policy))
    value_loss = jnp.mean((value - test_value) ** 2)

    print(f"Final evaluation policy loss {policy_loss:.5f} value loss: {test_loss:.5f}")

    print(f'count {(~(jnp.argmax(policy) == jnp.argmax(test_policy))).sum()}')
    print(policy_loss)

    print(value_loss)

    assert policy_loss < .52
    # assert jnp.sum(jnp.argmax(policy, -1) == jnp.argmax(test_policy, -1)) < 5
    assert value_loss < 0.02


def test_checkpointing(env, model):
    import numpy as np
    import shutil

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_env = jax.random.split(rng_key)
    state = env.init(rng_env)
    obs = flatten_pytree_batched(state.observation)
    policy, value = model(obs)

    first_checkpoint = Path('./local_checkpoint')
    if first_checkpoint.exists():
        shutil.rmtree(str(first_checkpoint))

    _, state = nnx.split(model)
    train.save_checkpoint(model, first_checkpoint.absolute())
    model = train.load_checkpoint(model, first_checkpoint.absolute())
    _, restored_state = nnx.split(model)
    jax.tree.map(np.testing.assert_array_equal, state, restored_state)

    rest_policy, rest_value = model(obs)

    np.testing.assert_array_equal(policy, rest_policy)
    np.testing.assert_array_equal(value, rest_value)
