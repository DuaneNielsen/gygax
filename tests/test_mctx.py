import jax
import jax.numpy as jnp
from train import vmap_flatten, MLP, get_recurrent_function
import dnd5e
from flax import nnx


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
        expected_discount = jnp.where(prev_state.current_player == state.current_player, 1.,  -1)
        assert jnp.all(context.discount == expected_discount)
        action = jnp.argmax(context.prior_logits, -1)

