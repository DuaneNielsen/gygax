import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class LiveProbabilityPlot:
    def __init__(self, num_probabilities=100):
        """
        Initialize the live probability plot.

        Args:
            num_probabilities (int): Number of probabilities to display
        """
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.num_probabilities = num_probabilities

        # Initialize data
        self.probabilities = np.zeros(num_probabilities)
        self.y_positions = np.arange(num_probabilities)

        # Create initial segments for lines
        segments = [[[0, y], [p, y]] for y, p in zip(self.y_positions, self.probabilities)]

        # Create line collection
        self.lines = LineCollection(segments, colors='gray', alpha=0.5)
        self.ax.add_collection(self.lines)

        # Create scatter points
        self.points = self.ax.scatter(self.probabilities, self.y_positions, color='blue', alpha=0.6, s=100)

        # Setup the rest of the plot
        self.ax.set_xlabel('Probability', fontsize=12)
        self.ax.set_ylabel('Index', fontsize=12)
        self.ax.set_title('Live Distribution of Probabilities', fontsize=14, pad=20)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlim(-1.05, 1.05)
        self.ax.set_ylim(-1, num_probabilities)
        self.reference_line = self.ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, new_probabilities):
        """
        Update the plot with new probability values.

        Args:
            new_probabilities: Array of new probability values
        """
        # Clear previous annotations
        for txt in self.ax.texts:
            txt.remove()

        # Update line segments
        segments = [[[0, y], [p, y]] for y, p in zip(self.y_positions, new_probabilities)]
        self.lines.set_segments(segments)

        # Update scatter points
        self.points.set_offsets(np.column_stack((new_probabilities, self.y_positions)))

        # Update annotations
        for i, prob in enumerate(new_probabilities):
            self.ax.text(prob + 0.02, i, f'{prob:.3f}', verticalalignment='center', fontsize=8)

        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# Example usage:
if __name__ == "__main__":
    import time

    # Create the plot
    plot = LiveProbabilityPlot(num_probabilities=100)

    # Update it in a loop
    try:
        while True:
            # Generate new random probabilities
            new_probs = np.random.random(100)

            # Update the plot
            plot.update(new_probs)

            # Wait a bit
            time.sleep(0.5)  # Update every 500ms

    except KeyboardInterrupt:
        plt.close('all')