import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from abc import ABC
from matplotlib.pyplot import axes
from typing import Protocol, Optional
from dataclasses import dataclass

import actions
from conditions import Conditions
from constants import *
from default_config import default_config

ACTION_NAMES = ['End Turn', 'Move', 'Melee Attack', 'Off-Hand', 'Ranged']
N_ACTIONS = len(ACTION_NAMES)
N_CHARS = 4
TOTAL_TARGETS = 8  # 4 ally + 4 enemy slots


def get_character_names(state, player):
    names = list(default_config[ConfigItems.PARTY][player].keys())
    return [names[p] for p in state.scene.party.pos[player]]


class Plot(ABC):
    ax: axes

    def refresh(self, data):
        pass


class HistogramPlot(Plot):
    def __init__(self, ax, state, player, data_key: str, title: str, xticklabels=None, character_names=None,
                 annotate_cells=False, sum=False, scale_min=0):
        self.ax = ax
        self.sum = sum
        self.scale_min = scale_min
        self.data_key = data_key
        self.player = player

        data = self.data(state)

        custom_cmap = sns.color_palette("Blues", as_cmap=True)
        custom_cmap.set_under('white')

        vmin = 0.001 if data.max() > 0 else 0
        vmax = data.max() if sum else 1

        sns.heatmap(data, annot=annotate_cells, fmt='d',
                    xticklabels=xticklabels,
                    yticklabels=character_names,
                    ax=ax, cmap=custom_cmap,
                    cbar_kws={'label': title},
                    cbar=False,
                    vmin=vmin,
                    vmax=vmax)

        ax.set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    def data(self, state):
        data = jnp.int32(state.observation.party[self.data_key][self.player])
        if self.sum:
            data = data.sum(-1) + self.scale_min
        return data

    def refresh(self, state):
        data = self.data(state)
        character_names = get_character_names(state, (actions.item() + self.player) % 2)

        heatmap = self.ax.collections[0]

        vmin = 0.001 if data.max() > 0 else 0
        vmax = data.max() if self.sum else 1

        heatmap.set_array(data.flatten())
        heatmap.set_clim(vmin, vmax)

        self.ax.set_yticklabels(character_names, rotation=0)

        for i, text in enumerate(self.ax.texts):
            row = i // data.shape[1]
            col = i % data.shape[1]
            text.set_text(f'{int(data[row, col])}')


class PositionPlot(Plot):
    def __init__(self, ax, positions):
        self.ax = ax
        grid = positions.reshape(1, -1)

        ax.matshow(grid, cmap='viridis')
        for i in range(grid.shape[1]):
            ax.text(i, 0, f'{int(grid[0, i])}',
                    ha='center', va='center',
                    color='white', fontsize=12)

        ax.set_title('Character Positions')
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([f'Char {i + 1}' for i in range(grid.shape[1])])
        ax.set_yticks([])

    def refresh(self, positions):
        grid = positions.reshape(1, -1)
        im = self.ax.get_images()[0]
        im.set_data(grid)

        for i, txt in enumerate(self.ax.texts):
            txt.set_text(f'{int(grid[0, i])}')

        return im


class CharacterDataVisualizer:
    def __init__(self, fig, gridspec, state):
        self.fig = fig
        self.gs = gridspec
        self.state = state
        self.left_plots = {}
        self.right_plots = {}
        self.left_pos = None
        self.right_pos = None

    def create_visualization(self):
        current_player = actions.item()
        enemy_player = (current_player + 1) % 2

        self.left_pos = PositionPlot(self.fig.add_subplot(self.gs[0, 0]),
                                     self.state.scene.party.pos[current_player])
        self.right_pos = PositionPlot(self.fig.add_subplot(self.gs[0, 2]),
                                      self.state.scene.party.pos[enemy_player])

        self.left_plots = self._create_party_plots(self.state, current_player)
        self.right_plots = self._create_party_plots(self.state, enemy_player, grid_offset=2)

    def _create_party_plots(self, state, player, grid_offset=0):
        plots = {}
        player_names = get_character_names(state, player)

        plots['hitpoints'] = HistogramPlot(
            self.fig.add_subplot(self.gs[0, 1 + grid_offset]),
            state, player,
            data_key='hitpoints',
            title='hitpoints',
            xticklabels=list(range(HP_LOWER, HP_UPPER)),
            character_names=player_names,
        )
        plots['armor_class'] = HistogramPlot(
            self.fig.add_subplot(self.gs[1, 0 + grid_offset]),
            state, player,
            data_key='armor_class',
            title='armor_class',
            xticklabels=list(range(AC_LOWER, AC_UPPER)),
            character_names=player_names
        )
        plots['ability_modifier'] = HistogramPlot(
            self.fig.add_subplot(self.gs[1, 1 + grid_offset]),
            state, player,
            data_key='ability_modifier',
            sum=True,
            scale_min=ABILITY_MODIFIER_LOWER,
            title='ability modifiers',
            xticklabels=[c.name for c in Abilities],
            character_names=player_names,
            annotate_cells=True,
        )
        plots['action_resources'] = HistogramPlot(
            self.fig.add_subplot(self.gs[2, 0 + grid_offset]),
            state, player,
            data_key='action_resources',
            sum=True,
            title='action resources',
            xticklabels=[c.name for c in ActionResourceType],
            character_names=player_names,
            annotate_cells=True,
        )
        plots['conditions'] = HistogramPlot(
            self.fig.add_subplot(self.gs[2, 1 + grid_offset]),
            state, player,
            data_key='conditions',
            sum=True,
            title='conditions',
            xticklabels=[c.name for c in Conditions],
            character_names=player_names,
            annotate_cells=True,
        )
        return plots

    def refresh(self, state):
        self.state = state
        current_player = actions.item()
        enemy_player = (current_player + 1) % 2

        self.left_pos.refresh(state.scene.party.pos[current_player])
        self.right_pos.refresh(state.scene.party.pos[enemy_player])

        for plot in self.left_plots.values():
            plot.refresh(state)

        for plot in self.right_plots.values():
            plot.refresh(state)


@dataclass
class ActionSelection:
    source_char: int
    action: int
    target_party: int
    target_slot: int

    @property
    def encoded(self) -> int:
        """Convert to flat action index"""
        return np.ravel_multi_index(
            [self.source_char, self.action, self.target_party, self.target_slot],
            [N_CHARS, N_ACTIONS, N_PLAYERS, N_CHARS]
        )

    @staticmethod
    def decode(action_idx: int) -> 'ActionSelection':
        """Create from flat action index"""
        char_idx, action, target_party, target_slot = np.unravel_index(
            action_idx, [N_CHARS, N_ACTIONS, 2, N_CHARS]
        )
        return ActionSelection(char_idx, action, target_party, target_slot)


class ActionPolicy(Protocol):
    """Interface for action selection policies"""

    def select_action(self, state) -> ActionSelection:
        """Select an action given the current state"""
        pass

    def get_action_probs(self, state) -> np.ndarray:
        """Get probability distribution over actions"""
        pass


class HumanPolicy(ActionPolicy):
    """Policy that delegates action selection to human via UI"""

    def __init__(self):
        self.last_action: Optional[ActionSelection] = None
        self.waiting_for_input = True

    def select_action(self, state) -> ActionSelection:
        self.waiting_for_input = True
        while self.waiting_for_input:
            plt.pause(0.1)  # Allow UI to update
        action = self.last_action
        self.last_action = None
        return action

    def get_action_probs(self, state) -> np.ndarray:
        """Return uniform distribution over legal actions"""
        return np.ones((N_CHARS, N_ACTIONS, 2, N_CHARS)) / (N_CHARS * N_ACTIONS * 2 * N_CHARS)

    def notify_action(self, action: ActionSelection):
        """Called by UI when human selects an action"""
        self.last_action = action
        self.waiting_for_input = False


class ActionGrid:
    """Manages the interactive action selection grid"""

    def __init__(self, fig: plt.Figure, gridspec: gridspec.GridSpec,
                 row: int, policy: ActionPolicy):
        self.fig = fig
        self.gs = gridspec
        self.row = row
        self.policy = policy
        self.buttons: List[List[List[plt.Button]]] = []
        self.axs: List[plt.Axes] = []

    def create_grid(self, state):
        """Create the action selection grid"""
        self.axs = [self.fig.add_subplot(self.gs[self.row, i]) for i in range(N_CHARS)]
        legal_actions = self._get_legal_action_mask(state)

        for char_idx, ax in enumerate(self.axs):
            name = get_character_names(state, 0)[char_idx]
            self._create_character_grid(name, char_idx, ax, legal_actions)
            ax.set_xticks([])
            ax.set_yticks([])

        self._update_button_colors(state)

    def _create_character_grid(self, name, char_idx: int, ax: plt.Axes, legal_actions):
        """Create grid for a single character"""
        ax.set_title(name)

        button_height = 0.08
        button_width = 1 / TOTAL_TARGETS

        if len(self.buttons) <= char_idx:
            self.buttons.append([])

        # Add action labels
        for i, action in enumerate(ACTION_NAMES):
            ax.text(-0.1, 1 - (i + 0.5) * button_height,
                    action, ha='right', va='center', transform=ax.transAxes)

        # Create buttons
        for i in range(N_ACTIONS):
            self.buttons[char_idx].append([])
            for j in range(TOTAL_TARGETS):
                target_party = 1 if j >= N_CHARS else 0
                is_legal = legal_actions[char_idx, i, target_party, j % N_CHARS]

                btn_x = j / TOTAL_TARGETS
                btn_y = 1 - ((i + 0.5) * button_height)

                btn = plt.Button(ax.inset_axes([btn_x, btn_y, button_width, button_height]),
                                 '', color='lightgray')

                action = ActionSelection(char_idx, i, target_party, j % N_CHARS)
                btn.on_clicked(lambda _, a=action: self._on_button_click(a))

                self.buttons[char_idx][i].append(btn)

    def _on_button_click(self, action: ActionSelection):
        """Handle button clicks by notifying policy"""
        if isinstance(self.policy, HumanPolicy):
            self.policy.notify_action(action)

    def _get_legal_action_mask(self, state) -> np.ndarray:
        """Get mask of legal actions"""
        return state.legal_action_mask.reshape(N_CHARS, N_ACTIONS, 2, N_CHARS)

    def _update_button_colors(self, state):
        """Update button colors based on legal actions and policy probabilities"""
        legal_actions = self._get_legal_action_mask(state)
        probs = self.policy.get_action_probs(state)

        for char_idx in range(N_CHARS):
            for i in range(N_ACTIONS):
                for j in range(TOTAL_TARGETS):
                    target_party = 1 if j >= N_CHARS else 0
                    target_slot = j % N_CHARS

                    button = self.buttons[char_idx][i][j]
                    is_legal = legal_actions[char_idx, i, target_party, target_slot]
                    prob = probs[char_idx, i, target_party, target_slot]

                    # Base color on probability for legal actions
                    if is_legal:
                        intensity = int(255 * prob)
                        color = f'#{intensity:02x}{intensity:02x}ff'  # Blue with varying intensity
                    else:
                        color = 'lightgray'

                    button.color = color
                    button.ax.set_facecolor(color)
                    button.set_active(is_legal)

    def refresh(self, state):
        """Update grid for new state"""
        names = get_character_names(state, actions.item())
        for char_idx, ax in enumerate(self.axs):
            ax.set_title(names[char_idx])
        self._update_button_colors(state)


class PartyVisualizer:
    """Main visualization class coordinating character data and action grid"""

    def __init__(self, env, state, party_idx=0, policy: Optional[ActionPolicy] = None):
        self.step = env.step
        self.state = state
        self.party_name = "Player Characters" if party_idx == 0 else "NPCs"
        self.fig = None
        self.character_visualizer = None
        self.action_grid = None
        self.policy = policy or HumanPolicy()

    def visualize(self, state):
        """Create complete visualization"""
        self.fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(4, 4, figure=self.fig, height_ratios=[1, 1, 1, 1.5])
        self.fig.suptitle(f'D&D 5e {self.party_name} State', size=16)

        # Create character visualization
        self.character_visualizer = CharacterDataVisualizer(self.fig, gs, state)
        self.character_visualizer.create_visualization()

        # Create action grid
        self.action_grid = ActionGrid(self.fig, gs, 3, self.policy)
        self.action_grid.create_grid(state)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05,
                            hspace=0.5, wspace=0.3)

        return self.fig

    def refresh_state(self, state):
        """Update visualization for new state"""
        self.state = state
        self.character_visualizer.refresh(state)
        self.action_grid.refresh(state)
        plt.draw()


if __name__ == '__main__':
    import jax
    import dnd5e

    # Example with human policy (interactive)
    env = dnd5e.DND5E()
    env = dnd5e.wrap_win_first_death(env)
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    jit_init = jax.jit(env.init, backend='cpu')
    jit_step = jax.jit(env.step, backend='cpu')
    state = jit_init(rng_init)

    # Example with custom policy
    class RandomPolicy(ActionPolicy):
        def select_action(self, state):
            legal_mask = state.legal_action_mask.reshape(N_CHARS, N_ACTIONS, 2, N_CHARS)
            legal_actions = np.argwhere(legal_mask)
            idx = np.random.randint(len(legal_actions))
            return ActionSelection(*legal_actions[idx])

        def get_action_probs(self, state):
            legal_mask = state.legal_action_mask.reshape(N_CHARS, N_ACTIONS, 2, N_CHARS)
            probs = legal_mask.astype(float)
            probs /= probs.sum()
            return probs


    visualizer = PartyVisualizer(env, state, party_idx=0, policy=HumanPolicy())
    fig = visualizer.visualize(state)

    random_policy = RandomPolicy()

    from train import MLP, load_checkpoint, get_recurrent_function
    from tree_serialization import flatten_pytree_batched
    import jax
    from flax import nnx
    import mctx

    rng_key, rng_env, rng_model = jax.random.split(jax.random.PRNGKey(0), 3)
    dummy_state = env.init(rng_env)
    dummy_obs = flatten_pytree_batched(dummy_state.observation)
    observation_features = dummy_obs.shape[0]
    baseline_model = MLP(observation_features, 128, env.num_actions, rngs=nnx.Rngs(rng_model))
    baseline_chkpt = '/mnt/megatron/data/dnd5e/wandb/offline-run-20241124_222111-6p2bjtub/files/model_epoch199'
    baseline_model = load_checkpoint(baseline_model, baseline_chkpt)

    recurrent_fn = get_recurrent_function(env)


    def policy(state, rng_policy):

        obs = flatten_pytree_batched(state.observation)
        logits, value = baseline_model(obs[None, ...])
        value = value.squeeze(-1)
        state = jax.tree.map(lambda s: s[None, ...], state)
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=baseline_model,
            rng_key=rng_policy,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=64,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0
        )
        return policy_output.action.squeeze(0)

    policy = jax.jit(policy, backend='cpu')

    turn = 0
    while True:
        rng_policy = jax.random.fold_in(rng_key, 0)
        if state.current_player == 0:
            action = visualizer.policy.select_action(state).encoded
            decoded_action = dnd5e.decode_action(action, state.current_player, state.scene.party.pos)
            print(dnd5e.repr_action(decoded_action))
            state = jit_step(state, action)
            visualizer.refresh_state(state)
        else:
            # action = random_policy.select_action(state).encoded
            action = policy(state, rng_key)
            decoded_action = dnd5e.decode_action(action, state.current_player, state.scene.party.pos)
            print(dnd5e.repr_action(decoded_action))
            state = jit_step(state, action)
            visualizer.refresh_state(state)
        turn += 1
        if state.terminated:
            rng_key, rng_env = jax.random.split(rng_policy, 2)
            state = jit_init(rng_env)
            visualizer.refresh_state(state)