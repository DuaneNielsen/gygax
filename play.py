import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from constants import N_ACTIONS, Actions, N_PLAYERS, N_CHARACTERS
import matplotlib.gridspec as gridspec

ACTION_NAMES = ['End Turn', 'Move', 'Melee Attack', 'Off-Hand', 'Ranged']
N_ACTIONS = len(ACTION_NAMES)
N_CHARS = 4
TOTAL_TARGETS = 8  # 4 ally + 4 enemy slots

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
                        self.refresh_state()
                        plt.draw()
                        return  # Exit after processing the button click

    def refresh_action_grid(self, axs):
        for char_idx, ax in enumerate(axs):
            for j in range(TOTAL_TARGETS):
                for i in range(N_ACTIONS):
                    button = self.callback_data['buttons'][char_idx][i][j]
                    target_party = 1 if j >= N_CHARS else 0
                    is_legal = self.legal_action_mask[char_idx, i, target_party, j % N_CHARS]

                    party_color = 'lightblue' if self.state.current_player == 0 else 'red'
                    button.color = 'lightgray' if not is_legal else party_color
                    button.ax.set_facecolor(button.color)

                    button.set_active(is_legal)


    def refresh_state(self):

        self.refresh_positions(self.party0_positions_ax, 0)
        self.refresh_hitpoints(self.party0_hitpoint_ax, 0)
        self.refresh_armor_class(self.party0_armor_class, 0)
        self.refresh_ability_modifiers(self.party0_ability_modifiers, 0)
        self.refresh_action_resources(self.party0_action_resources, 0)
        self.refresh_conditions(self.party0_conditions, 0)

        self.refresh_positions(self.party1_positions_ax, 1)
        self.refresh_hitpoints(self.party1_hitpoint_ax, 1)
        self.refresh_armor_class(self.party1_armor_class, 1)
        self.refresh_ability_modifiers(self.party1_ability_modifiers, 1)
        self.refresh_action_resources(self.party1_action_resources, 1)
        self.refresh_conditions(self.party1_conditions, 1)


    def create_grid(self, fig, char_idx, ax, legal_actions):
        ax.set_title(f'Character {char_idx + 1}')

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

    def plot_positions(self, ax, party_idx):
        """Plot character positions in a 1x4 grid"""
        positions = self.party.pos[party_idx]
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
        return ax

    def plot_hitpoints(self, ax, party_idx):
        """Plot character hit points"""
        hitpoints = self.party.hitpoints[party_idx]
        x = np.arange(len(hitpoints))
        bars = ax.bar(x, hitpoints, color='lightgreen')

        ax.set_title('Hit Points')
        ax.set_xlabel('Character')
        ax.set_ylabel('HP')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Char {i + 1}' for i in x])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        return ax

    def plot_armor_class(self, ax, party_idx):
        """Plot character armor class"""
        ac = self.party.armor_class[party_idx]
        x = np.arange(len(ac))
        bars = ax.bar(x, ac, color='lightblue')

        ax.set_title('Armor Class')
        ax.set_xlabel('Character')
        ax.set_ylabel('AC')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Char {i + 1}' for i in x])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        return ax

    def plot_ability_modifiers(self, ax, party_idx):
        """Plot ability modifiers table"""
        ability_labels = ['STR', 'DEX', 'CON', 'INT', 'WIS', 'CHA']
        modifiers = self.party.ability_modifier[party_idx]

        sns.heatmap(modifiers, annot=True, fmt='d',
                    xticklabels=ability_labels,
                    yticklabels=[f'Char {i + 1}' for i in range(modifiers.shape[0])],
                    ax=ax, cmap='RdYlBu', center=0,
                    cbar_kws={'label': 'Modifier'})

        ax.set_title('Ability Modifiers')
        return ax

    def refresh_positions(self, ax, party_idx):
        """Refresh character positions plot"""
        positions = self.party.pos[party_idx]
        grid = positions.reshape(1, -1)  # Force 1x4 layout

        # Update grid data
        im = ax.get_images()[0]
        im.set_data(grid)

        # Update text values
        for i, txt in enumerate(ax.texts):
            txt.set_text(f'{int(grid[0, i])}')

        return im

    def refresh_hitpoints(self, ax, party_idx):
        """Refresh hit points bar plot"""
        hitpoints = self.party.hitpoints[party_idx]

        # Update bar heights
        bars = ax.containers[0]
        for bar, hp in zip(bars, hitpoints):
            bar.set_height(hp)

        # Update value labels
        for i, (bar, txt) in enumerate(zip(bars, ax.texts)):
            txt.set_text(f'{int(hitpoints[i])}')
            txt.set_position((bar.get_x() + bar.get_width() / 2., hitpoints[i]))

        # Update y-axis limit if needed
        ax.relim()
        ax.autoscale_view(scalex=False)

        return bars

    def refresh_armor_class(self, ax, party_idx):
        """Refresh armor class bar plot"""
        ac = self.party.armor_class[party_idx]

        # Update bar heights
        bars = ax.containers[0]
        for bar, val in zip(bars, ac):
            bar.set_height(val)

        # Update value labels
        for i, (bar, txt) in enumerate(zip(bars, ax.texts)):
            txt.set_text(f'{int(ac[i])}')
            txt.set_position((bar.get_x() + bar.get_width() / 2., ac[i]))

        # Update y-axis limit if needed
        ax.relim()
        ax.autoscale_view(scalex=False)

        return bars

    def refresh_ability_modifiers(self, ax, party_idx):
        """Refresh ability modifiers heatmap"""
        modifiers = self.party.ability_modifier[party_idx]

        # Update heatmap data
        heatmap = ax.collections[0]
        heatmap.set_array(modifiers.flatten())

        # Update annotations
        for i, text in enumerate(ax.texts):
            row = i // modifiers.shape[1]
            col = i % modifiers.shape[1]
            text.set_text(f'{int(modifiers[row, col])}')

        # Update colorbar scale if needed
        vmin, vmax = modifiers.min(), modifiers.max()
        heatmap.set_clim(vmin, vmax)

        # Update colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))
        cbar._draw_all()

        # Redraw the plot
        ax.figure.canvas.draw()

    def plot_action_resources(self, ax, party_idx):
        """Plot action resources table"""
        resources = self.party.action_resources[party_idx]

        sns.heatmap(resources, annot=True, fmt='d',
                    xticklabels=[f'Res {i + 1}' for i in range(resources.shape[1])],
                    yticklabels=[f'Char {i + 1}' for i in range(resources.shape[0])],
                    ax=ax, cmap='YlOrRd',
                    cbar_kws={'label': 'Resources'})

        ax.set_title('Action Resources')
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        return ax

    def refresh_action_resources(self, ax, party_idx):
        """Refresh action resources heatmap"""
        resources = self.party.action_resources[party_idx]

        # Update heatmap data
        heatmap = ax.collections[0]
        heatmap.set_array(resources.flatten())

        # Update annotations
        for i, text in enumerate(ax.texts):
            row = i // resources.shape[1]
            col = i % resources.shape[1]
            text.set_text(f'{int(resources[row, col])}')

        # Update colorbar scale if needed
        vmin, vmax = resources.min(), resources.max()
        heatmap.set_clim(vmin, vmax)

        # Update colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))
        cbar._draw_all()

        # Redraw the plot
        ax.figure.canvas.draw()

    import seaborn as sns
    import numpy as np

    def plot_conditions(self, ax, party_idx):
        """Plot conditions table"""
        conditions = self.party.conditions[party_idx]

        sns.heatmap(conditions, annot=True, fmt='d',
                    xticklabels=[f'Cond {i + 1}' for i in range(conditions.shape[1])],
                    yticklabels=[f'Char {i + 1}' for i in range(conditions.shape[0])],
                    ax=ax, cmap='Reds',
                    cbar_kws={'label': 'Stacks'})

        ax.set_title('Conditions')
        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        return ax

    def refresh_conditions(self, ax, party_idx):
        """Refresh conditions heatmap"""
        conditions = self.party.conditions[party_idx]

        # Update heatmap data
        heatmap = ax.collections[0]
        heatmap.set_array(conditions.flatten())

        # Update annotations
        for i, text in enumerate(ax.texts):
            row = i // conditions.shape[1]
            col = i % conditions.shape[1]
            text.set_text(f'{int(conditions[row, col])}')

        # Update colorbar scale if needed
        vmin, vmax = conditions.min(), conditions.max()
        heatmap.set_clim(vmin, vmax)

        # Update colorbar
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.linspace(vmin, vmax, num=5))
        cbar._draw_all()

        # Redraw the plot
        ax.figure.canvas.draw()

    def visualize(self, state):
        """Create main visualization with character stats and action grid in one figure."""
        fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1.5])
        fig.suptitle(f'D&D 5e {self.party_name} State', size=16)

        # Top rows for character stats
        self.party0_positions_ax = self.plot_positions(fig.add_subplot(gs[0, 0]), 0)
        self.party0_hitpoint_ax = self.plot_hitpoints(fig.add_subplot(gs[0, 1]), 0)
        self.party0_armor_class = self.plot_armor_class(fig.add_subplot(gs[1, 0]), 0)
        self.party0_ability_modifiers = self.plot_ability_modifiers(fig.add_subplot(gs[1, 1]), 0)
        self.party0_action_resources = self.plot_action_resources(fig.add_subplot(gs[2, 0]), 0)
        self.party0_conditions = self.plot_conditions(fig.add_subplot(gs[2, 1]), 0)

        self.party1_positions_ax = self.plot_positions(fig.add_subplot(gs[0, 2]), 1)
        self.party1_hitpoint_ax = self.plot_hitpoints(fig.add_subplot(gs[0, 3]), 1)
        self.party1_armor_class = self.plot_armor_class(fig.add_subplot(gs[1, 2]), 1)
        self.party1_ability_modifiers = self.plot_ability_modifiers(fig.add_subplot(gs[1, 3]), 1)
        self.party1_action_resources = self.plot_action_resources(fig.add_subplot(gs[2, 2]), 1)
        self.party1_conditions = self.plot_conditions(fig.add_subplot(gs[2, 3]), 1)

        # Bottom row for action grid, with each character in its own column
        action_axs = [fig.add_subplot(gs[3, i]) for i in range(N_CHARS)]
        self.callback_data['legal_actions'] = self.legal_action_mask
        self.create_action_grid(fig, self.legal_action_mask, action_axs)

        # Manually adjust layout to make space for buttons and ensure proper alignment
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05, hspace=0.3, wspace=0.3)

        self.callback_data['fig'] = fig
        return fig


if __name__ == '__main__':
    import jax
    import dnd5e

    env = dnd5e.DND5E()
    rng, rng_init = jax.random.split(jax.random.PRNGKey(0), 2)
    state = env.init(rng_init)

    visualizer = PartyVisualizer(env, state, party_idx=0)  # Show PCs
    fig = visualizer.visualize(state)

    plt.show()
