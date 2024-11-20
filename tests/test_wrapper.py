import pytest
import jax
import functools
from typing import Any
from pgx._src.struct import dataclass

# Assuming these are imported or defined in your actual implementation
import pgx.core as core
from wrappers import DND5EProxy  # Import the proxy class
from jax import Array
import jax.numpy as jnp
from constants import N_CHARACTERS, N_PLAYERS, N_ACTIONS, FALSE

@dataclass
class MockState(core.State):
    def env_id(self):
        return "mock_env"


@dataclass
class State:
    # dnd5e specific
    _step_count: Array = jnp.zeros((1,), dtype=jnp.int32)
    current_player: Array = jnp.zeros(1)
    legal_action_mask: Array = jnp.zeros((N_PLAYERS, N_CHARACTERS, N_ACTIONS))
    observation : Array = jnp.zeros((3, 3))
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE



# Mock environment class for testing
class MockDND5E(core.Env):
    def __init__(self):
        super().__init__()
        self.init_calls = 0
        self.step_calls = 0
        self.observe_calls = 0

    def _init(self, key: jax.random.PRNGKey, config=None) -> core.State:
        self.init_calls += 1
        # Return a mock state
        return MockState(
            _step_count=jnp.array([0], dtype=jnp.int32),
            current_player=jnp.array([0], dtype=jnp.int32),
            legal_action_mask=jax.numpy.ones(9, dtype=bool),
            rewards=jax.numpy.zeros(2),
            terminated=FALSE,
            truncated=FALSE,
            observation=jax.numpy.zeros((3, 3))
        )

    def _step(self, state: core.State, action: jax.Array, key=None) -> core.State:
        self.step_calls += 1
        # Return a modified mock state
        return state.replace(
            current_player=1 - state.current_player,
            rewards=jax.numpy.ones(2) * 0.1
        )

    def _observe(self, state: core.State, player_id: jax.Array) -> jax.Array:
        self.observe_calls += 1
        return jax.numpy.zeros((3, 3))

    @property
    def id(self) -> core.EnvId:
        return "mock_env"

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


class TestDND5EProxy:
    @pytest.fixture
    def mock_env(self):
        """Fixture to create a fresh mock environment for each test"""
        return MockDND5E()

    def test_default_proxy_delegates_methods(self, mock_env):
        """Test that default proxy delegates methods without modification"""
        proxy = DND5EProxy(base_env=mock_env)

        # Generate a random key
        key = jax.random.PRNGKey(0)

        # Test init
        state = proxy.init(key)
        assert mock_env.init_calls == 1
        assert isinstance(state, core.State)

        # Test step
        new_state = proxy.step(state, action=jax.numpy.int32(4))
        assert mock_env.step_calls == 1
        assert isinstance(new_state, core.State)

        # Test observe
        observation = proxy.observe(new_state, player_id=jax.numpy.int32(0))
        assert mock_env.observe_calls == 3
        assert isinstance(observation, jax.Array)

    def test_init_wrapper(self, mock_env):
        """Test wrapping the init method"""

        def init_wrapper(base_env, key):
            # Add a side effect
            base_env.wrapper_init_called = True
            return base_env.init(key)

        proxy = DND5EProxy(base_env=mock_env, init_wrapper=init_wrapper)

        # Generate a random key
        key = jax.random.PRNGKey(0)

        # Call init
        state = proxy.init(key)

        # Check that both original and wrapper methods were called
        assert mock_env.init_calls == 1
        assert hasattr(mock_env, 'wrapper_init_called')
        assert mock_env.wrapper_init_called is True

    def test_step_wrapper(self, mock_env):
        """Test wrapping the step method"""

        def step_wrapper(base_env, state, action, key=None):
            # Add a side effect
            base_env.wrapper_step_called = True
            return base_env.step(state, action, key)

        proxy = DND5EProxy(base_env=mock_env, step_wrapper=step_wrapper)

        # Generate a random key and initial state
        key = jax.random.PRNGKey(0)
        initial_state = proxy.init(key)

        # Call step
        new_state = proxy.step(initial_state, action=jax.numpy.int32(4))

        # Check that both original and wrapper methods were called
        assert mock_env.step_calls == 1
        assert hasattr(mock_env, 'wrapper_step_called')
        assert mock_env.wrapper_step_called is True

    def test_observe_wrapper(self, mock_env):
        """Test wrapping the observe method"""

        def observe_wrapper(base_env, state, player_id):
            # Add a side effect
            base_env.wrapper_observe_called = True
            return base_env.observe(state, player_id)

        proxy = DND5EProxy(base_env=mock_env, observe_wrapper=observe_wrapper)

        # Generate a random key and initial state
        key = jax.random.PRNGKey(0)
        initial_state = proxy.init(key)

        # Call observe
        observation = proxy.observe(initial_state, player_id=jax.numpy.int32(0))

        # Check that both original and wrapper methods were called
        assert mock_env.observe_calls == 2
        assert hasattr(mock_env, 'wrapper_observe_called')
        assert mock_env.wrapper_observe_called is True

    def test_property_delegation(self, mock_env):
        """Test that properties are correctly delegated"""
        proxy = DND5EProxy(base_env=mock_env)

        # Check that properties are delegated correctly
        assert proxy.id == "mock_env"
        assert proxy.version == "v0"
        assert proxy.num_players == 2

    def test_multiple_wrappers(self, mock_env):
        """Test applying multiple wrappers simultaneously"""

        def init_wrapper(env, key):
            env.init_wrapper_called = True
            return env.init(key)

        def step_wrapper(env, state, action, key=None):
            env.step_wrapper_called = True
            return env.step(state, action, key)

        proxy = DND5EProxy(
            base_env=mock_env,
            init_wrapper=init_wrapper,
            step_wrapper=step_wrapper
        )

        # Generate a random key
        key = jax.random.PRNGKey(0)

        # Call init and step
        initial_state = proxy.init(key)
        new_state = proxy.step(initial_state, action=jax.numpy.int32(4))

        # Check that wrappers were called
        assert hasattr(mock_env, 'init_wrapper_called')
        assert hasattr(mock_env, 'step_wrapper_called')
        assert mock_env.init_wrapper_called is True
        assert mock_env.step_wrapper_called is True


# Helpful error handling test
def test_proxy_with_invalid_wrapper():
    """Test that an invalid wrapper raises an appropriate error"""
    mock_env = MockDND5E()

    with pytest.raises(TypeError):
        # Passing an invalid wrapper (not a callable)
        DND5EProxy(base_env=mock_env, init_wrapper="not a function")
