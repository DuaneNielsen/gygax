import functools
from typing import Callable, Optional
import jax
import pgx.core as core  # Assuming core is the module with the base Env class


class DND5EProxy:
    """
    A simple proxy wrapper for the DND5E environment that allows customization
    of init, step, and observe functions.
    """

    def __init__(
            self,
            base_env: core.Env,
            init_wrapper: Optional[Callable] = None,
            step_wrapper: Optional[Callable] = None,
            observe_wrapper: Optional[Callable] = None
    ):
        """
        Initialize the proxy with a base environment and optional wrapper functions.

        Args:
            base_env: The base environment to proxy
            init_wrapper: Optional wrapper for the init method
            step_wrapper: Optional wrapper for the step method
            observe_wrapper: Optional wrapper for the observe method
        """
        self._base_env = base_env

        if init_wrapper is not None:
            if not isinstance(init_wrapper, Callable):
                raise TypeError("init_wrapper should be type Callable")

        if step_wrapper is not None:
            if not isinstance(step_wrapper, Callable):
                raise TypeError("step_wrapper should be type Callable")

        if observe_wrapper is not None:
            if not isinstance(observe_wrapper, Callable):
                raise TypeError("observe_wrapper should be type Callable")

        default_init = lambda env, key: env.init(key)
        default_step = lambda env, state, action, key: env.step(state, action, key)
        default_observe = lambda env, state, player_id: env.observe(state, player_id)

        # Apply wrappers if provided, otherwise use original methods
        self._init_method = init_wrapper if init_wrapper is not None else default_init
        self._step_method = step_wrapper if step_wrapper is not None else default_step
        self._observe_method = observe_wrapper if observe_wrapper is not None else default_observe

    def init(self, key: jax.random.PRNGKey) -> core.State:
        """Wrapped initialization method matching DND5E signature"""
        return self._init_method(self._base_env, key)

    def step(self, state: core.State, action: jax.Array, key=None) -> core.State:
        """Wrapped step method matching DND5E signature"""
        return self._step_method(self._base_env, state, action, key)

    def observe(self, state: core.State, player_id: jax.Array) -> jax.Array:
        """Wrapped observe method matching DND5E signature"""
        return self._observe_method(self._base_env, state, player_id)

    # Delegate remaining methods and properties to base environment
    def __getattr__(self, name):
        """
        Dynamically delegate any other method or property
        to the base environment.
        """
        return getattr(self._base_env, name)