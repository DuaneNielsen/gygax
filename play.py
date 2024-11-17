import matplotlib.pyplot as plt
from matplotlib.pyplot import axes
import numpy as np
import seaborn as sns

import constants
from constants import *
from default_config import default_config
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
from abc import ABC

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
        # Binary for regular cells, autoscale for sums
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
        """Plot conditions table"""
        data = jnp.int32(state.observation.party[self.data_key][self.player])
        if self.sum:
            data = data.sum(-1) + self.scale_min
        return data

    def refresh(self, state):
        """Refresh conditions heatmap"""
        data = self.data(state)
        character_names = get_character_names(state, (state.current_player.item() + self.player) % 2)

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
        """Plot character positions in a 1x4 grid"""
        grid = positions.reshape(1, -1)  # Force 1x4 layout

        # Plot grid with integer values
        ax.matshow(grid, cmap='viridis')
        for i in range(grid.shape[1]):
            ax.text(i, 0, f'{int(grid[0, i])}',
                    ha='center', va='center',
                    color='white', fontsize=12)

        ax.set_title('Character Positions')
        ax.set_xticks(range(grid.shape[1]))
        ax.set_xticklabels([f'Char {i + 1}' for i in range(grid.shape[1])])
        ax.set_yticks([])

    def refresh_positions(self, positions):
        """Refresh character positions plot"""
        grid = positions.reshape(1, -1)  # Force 1x4 layout

        # Update grid data
        im = self.ax.get_images()[0]
        im.set_data(grid)

        # Update text values
        for i, txt in enumerate(self.ax.texts):
            txt.set_text(f'{int(grid[0, i])}')

        return im

        # Redraw the plot
        # ax.figure.canvas.draw()


class PartyVisualizer:
    def __init__(self, env, state, party_idx=0):
        self.step = env.step
        self.state = state
        self.party_name = "Player Characters" if party_idx == 0 else "NPCs"
        self.callback_data = {
            'buttons': [],
            'selected_action': None,
            'fig': None,
            'legal_actions': None,
        }
        self.fig = None

    @property
    def party(self):
        return self.state.scene.party

    @property
    def party_idx(self):
        return self.state.current_player.item()

    @property
    def legal_action_mask(self):
        return self.state.legal_action_mask.reshape(N_CHARACTERS, N_ACTIONS, N_PLAYERS, N_CHARACTERS)

    def create_target_label(self, target_idx):
        party = "Enemy" if target_idx >= N_CHARS else "Ally"
        slot = (target_idx % N_CHARS) + 1
        return f"{party}\n{slot}"

    def on_button_click(self, event):

        print('button clicked')

        # Ensure the click event is on an axes that contains a button
        if event.inaxes is None or not isinstance(event.inaxes, plt.Axes):
            return  # If not on a valid axes, exit

        # The button will be found by checking if the axes has buttons associated with it
        for button in self.callback_data['buttons']:
            for button_list in button:
                for btn in button_list:
                    if btn.ax == event.inaxes:
                        button_data = btn._button_data
                        source_char, action, target_slot = button_data['char'], button_data['action'], button_data[
                            'target']
                        target_party = 1 if target_slot >= N_CHARS else 0

                        # Encode the action
                        encoded_action = np.ravel_multi_index(
                            [source_char, action, target_party, target_slot % N_CHARS],
                            [N_CHARS, N_ACTIONS, 2, N_CHARS]
                        )

                        # Decode the action to confirm selection
                        decoded = np.unravel_index(encoded_action, [N_CHARS, N_ACTIONS, 2, N_CHARS])
                        print(f"Selected action: Character {decoded[0]}, Action {decoded[1]}, "
                              f"Target Party {decoded[2]}, Target Slot {decoded[3]}")

                        self.state = self.step(self.state, encoded_action)

                        self.refresh_action_grid(self.callback_data['axs'])  # Pass axs here
                        self.refresh_state(self.state)
                        plt.draw()
                        return  # Exit after processing the button click

    def refresh_action_grid(self, axs):
        names = get_character_names(self.state, self.state.current_player.item())
        for char_idx, ax in enumerate(axs):
            ax.set_title(names[char_idx])
            for j in range(TOTAL_TARGETS):
                for i in range(N_ACTIONS):
                    button = self.callback_data['buttons'][char_idx][i][j]
                    target_party = 1 if j >= N_CHARS else 0
                    is_legal = self.legal_action_mask[char_idx, i, target_party, j % N_CHARS]

                    party_color = 'lightblue' if self.state.current_player == 0 else 'red'
                    button.color = 'lightgray' if not is_legal else party_color
                    button.ax.set_facecolor(button.color)

                    button.set_active(is_legal)

    def refresh_state(self, state):

        current_player = state.current_player.item()
        enemy_player = (state.current_player.item() + 1) % 2

        party = dict(state.observation.party)

        self.left_pos.refresh(state.scene.party.pos[current_player])
        self.right_pos.refresh(state.scene.party.pos[enemy_player])

        for key, plot in self.left_plots.items():
            plot.refresh(state)

        for key, plot in self.right_plots.items():
            plot.refresh(state)

        plt.draw()

    def create_grid(self, fig, char_idx, ax, legal_actions):
        ax.set_title(get_character_names(self.state, 0)[char_idx])

        n_rows = N_ACTIONS
        n_cols = TOTAL_TARGETS
        button_height = 0.08
        button_width = 1 / n_cols  # Set width to fully occupy the column

        # Clear any existing buttons for this character in case of re-drawing
        if len(self.callback_data['buttons']) <= char_idx:
            self.callback_data['buttons'].append([])

        # Adjusting the y-position of action names
        for i, action in enumerate(ACTION_NAMES):
            ax.text(-0.1, 1 - (i + 0.5) * button_height,  # Center the text vertically
                    action, ha='right', va='center', transform=ax.transAxes)

        for i in range(N_ACTIONS):
            self.callback_data['buttons'][char_idx].append([])  # Initialize for each action
            for j in range(TOTAL_TARGETS):
                target_party = 1 if j >= N_CHARS else 0
                is_legal = legal_actions[char_idx, i, target_party, j % N_CHARS]

                # Position button within subplot bounds (relative positioning)
                btn_x = j / n_cols  # Position directly based on column index
                btn_y = 1 - ((i + 0.5) * button_height)  # Center button vertically

                # Place the button within the subplot
                btn = plt.Button(ax.inset_axes([btn_x, btn_y, button_width, button_height]),
                                 '', color='lightgray' if not is_legal else 'lightblue')

                btn._button_data = {'char': char_idx, 'action': i, 'target': j}
                btn.on_clicked(self.on_button_click)

                self.callback_data['buttons'][char_idx][i].append(btn)

    def create_action_grid(self, fig, legal_actions, axs):
        self.callback_data['buttons'] = []
        self.callback_data['axs'] = axs  # Store axs for later use
        for char_idx, ax in enumerate(axs):
            self.create_grid(fig, char_idx, ax, legal_actions)
            ax.set_xticks([])
            ax.set_yticks([])

    def visualize(self, state):
        """Create main visualization with character stats and action grid in one figure."""
        self.fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(4, 4, figure=self.fig, height_ratios=[1, 1, 1, 1.5])
        self.fig.suptitle(f'D&D 5e {self.party_name} State', size=16)

        party = state.observation.party
        current_player = state.current_player.item()
        enemy_player = (state.current_player.item() + 1) % 2

        self.left_pos = PositionPlot(self.fig.add_subplot(gs[0, 0]), state.scene.party.pos[current_player])
        self.right_pos = PositionPlot(self.fig.add_subplot(gs[0, 2]), state.scene.party.pos[enemy_player])

        def plot_party(state, player, grid_offset=0):
            plots = {}
            player_names = get_character_names(state, player)

            plots['hitpoints'] = HistogramPlot(
                self.fig.add_subplot(gs[0, 1 + grid_offset]),
                state,
                player,
                data_key='hitpoints',
                title='hitpoints',
                xticklabels=list(range(HP_LOWER, HP_UPPER)),
                character_names=player_names,
            )
            plots['armor_class'] = HistogramPlot(
                self.fig.add_subplot(gs[1, 0 + grid_offset]),
                state,
                player,
                data_key='armor_class',
                title='armor_class',
                xticklabels=list(range(AC_LOWER, AC_UPPER)),
                character_names=player_names
            )
            plots['ability_modifier'] = HistogramPlot(
                self.fig.add_subplot(gs[1, 1 + grid_offset]),
                state,
                player,
                data_key='ability_modifier',
                sum=True,
                scale_min=ABILITY_MODIFIER_LOWER,
                title='ability modifiers',
                xticklabels=[c.name for c in Abilities],
                character_names=player_names,
                annotate_cells=True,
            )
            plots['action_resources'] = HistogramPlot(
                self.fig.add_subplot(gs[2, 0 + grid_offset]),
                state,
                player,
                data_key='action_resources',
                sum=True,
                title='action resources',
                xticklabels=[c.name for c in ActionResourceType],
                character_names=player_names,
                annotate_cells=True,
            )

            plots['conditions'] = HistogramPlot(
                self.fig.add_subplot(gs[2, 1 + grid_offset]),
                state,
                player,
                data_key='conditions',
                sum=True,
                title='conditions',
                xticklabels=[c.name for c in Conditions],
                character_names=player_names,
                annotate_cells=True,
            )
            return plots

        self.left_plots = plot_party(state, current_player)
        self.right_plots = plot_party(state, enemy_player, 2)

        # Bottom row for action grid, with each character in its own column
        action_axs = [self.fig.add_subplot(gs[3, i]) for i in range(N_CHARS)]
        self.callback_data['legal_actions'] = self.legal_action_mask
        self.create_action_grid(self.fig, self.legal_action_mask, action_axs)

        # Manually adjust layout to make space for buttons and ensure proper alignment
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.5, wspace=0.3)

        self.callback_data['fig'] = self.fig
        return self.fig


if __name__ == '__main__':
    import jax
    import dnd5e

    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    visualizer = PartyVisualizer(env, state, party_idx=0)  # Show PCs
    fig = visualizer.visualize(state)

    plt.show()
