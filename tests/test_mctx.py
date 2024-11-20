import jax
import jax.numpy as jnp
from train import vmap_flatten, MLP, get_recurrent_function, make_selfplay
import dnd5e
from flax import nnx
import mctx
from pgx.experimental import act_randomly


def test_recurrent_function():
    batch_size = 2

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model, rng_env, rng_policy = jax.random.split(rng_key, 4)
    env = dnd5e.DND5E()
    recurrent_fn = get_recurrent_function(env)
    num_actions = env.num_actions

    state = jax.vmap(env.init)(jax.random.split(rng_env, batch_size))
    observation = vmap_flatten(state.observation)
    model = MLP(observation.shape[1], 128, num_actions, rngs=nnx.Rngs(rng_model))

    logits, value = jax.vmap(model)(observation)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    action = jnp.argmax(jax.nn.softmax(logits, -1), -1)

    for i in range(10):
        prev_state = jax.tree.map(lambda s: s.copy(), state)
        context, state = recurrent_fn(model, rng_key, action, state)
        assert jnp.all(state.terminated == False)
        expected_discount = jnp.where(prev_state.current_player == state.current_player, 1., -1)
        assert jnp.all(context.discount == expected_discount)
        action = jnp.argmax(context.prior_logits, -1)


def test_random_search():
    batch_size = 2

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_model, rng_env, rng_policy, rng_search = jax.random.split(rng_key, 5)
    env = dnd5e.DND5E()
    recurrent_fn = get_recurrent_function(env)
    num_actions = env.num_actions

    state = jax.vmap(env.init)(jax.random.split(rng_env, batch_size))
    observation = vmap_flatten(state.observation)
    model = MLP(observation.shape[1], 128, num_actions, rngs=nnx.Rngs(rng_model))

    logits, value = jax.vmap(model)(observation)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    root = mctx.RootFnOutput(prior_logits=logits, value=value.squeeze(-1), embedding=state)

    def root_action_selection_fn(rng_key, tree, node_index):
        action = act_randomly(rng_key, tree.embeddings.legal_action_mask[node_index])
        # jax.debug.print('{}', action)
        return action

    def interior_action_selection_fn(rng_key, tree, node_index, depth):
        action = act_randomly(rng_key, tree.embeddings.legal_action_mask[node_index])
        # jax.debug.print('{}', action)
        return action

    search_tree = mctx.search(params=model,
                              rng_key=rng_search,
                              root=root,
                              recurrent_fn=recurrent_fn,
                              root_action_selection_fn=root_action_selection_fn,
                              interior_action_selection_fn=interior_action_selection_fn,
                              num_simulations=20,
                              )

def test_selfplay_closure():
    batch_size = 2
    selfplay_batch_size = 2
    selfplay_max_steps = 10
    selfplay_num_simulations = 20
    num_devices = len(jax.local_devices())
    rng_env, rng_model, rng_search = jax.random.split(jax.random.PRNGKey(0), 3)
    env = dnd5e.DND5E()
    state = jax.vmap(env.init)(jax.random.split(rng_env, batch_size))
    observation = vmap_flatten(state.observation)
    model = MLP(observation.shape[1], 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    selfplay = make_selfplay(env, selfplay_batch_size, selfplay_max_steps, selfplay_num_simulations)

    rng_selfplay_devices = jax.random.split(rng_search, num_devices)
    data = selfplay(model, rng_selfplay_devices)

