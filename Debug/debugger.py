import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
from moviepy.editor import ImageSequenceClip
from multiprocessing import Pool, cpu_count
import numpy as np
import os
from tqdm import tqdm
from background import path_states


goal = (11, 9)

grid_size = (16, 24)
initial = (8, 10)
goal = (11, 9)
blocks = [
    (14, 9),
    (13, 10),
    (12, 10),
    (11, 10),
    (11, 12),
    (11, 13),
    (10, 9),
    (10, 10),
    (10, 12),
    (10, 13),
    (9, 9),
    (8, 9),
    (7, 9),
    (6, 9),
]


class GridPlotter:
    def __init__(self, grid_size=grid_size, initial=initial, goal=goal, blocks=blocks):
        self.grid_size = grid_size
        self.initial = initial
        self.goal = goal
        self.blocks = blocks
        self.path = []

    def set_path(self, path):
        self.path = path

    def plot_grid(self):
        fig, ax = plt.subplots()

        # Manually drawing the gridlines
        for x in range(self.grid_size[1] + 1):
            ax.plot([x, x], [0, self.grid_size[0]], color="black", linewidth=1)
        for y in range(self.grid_size[0] + 1):
            ax.plot([0, self.grid_size[1]], [y, y], color="black", linewidth=1)

        # Draw the black obstacles inside the grid
        for obstacle in self.blocks:
            rect = plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, facecolor="black")
            ax.add_patch(rect)

        # Mark the start and goal positions within the grid
        ax.text(
            self.initial[1] + 0.5,
            self.initial[0] + 0.5,
            "S",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            self.goal[1] + 0.5,
            self.goal[0] + 0.5,
            "G",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

        # Plot the path with a gradient of colors
        if self.path:
            colors = cm.rainbow(np.linspace(0, 1, len(self.path)))
            for i, (x, y) in enumerate(self.path):
                ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=colors[i]))
                ax.text(
                    y + 0.5,
                    x + 0.5,
                    f"{i+1}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        # Set labels for axes
        ax.set_xlabel("Y axis", fontsize=12, fontweight="bold")
        ax.set_ylabel("X axis", fontsize=12, fontweight="bold")

        # Set the ticks to align with the centers of squares
        ax.set_xticks(np.arange(0.5, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0.5, self.grid_size[0], 1))
        ax.set_xticklabels(np.arange(0, self.grid_size[1], 1))
        ax.set_yticklabels(np.arange(0, self.grid_size[0], 1))

        # Adjust the limits so that the ticks correspond to the little squares
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])

        # Show the plot
        plt.savefig()


class GridPlotterWithMultiprocessing(GridPlotter):
    def __init__(
        self,
        grid_size=grid_size,
        initial=initial,
        goal=goal,
        blocks=blocks,
        save_dir="./path",
    ):
        super().__init__(grid_size, initial, goal, blocks)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.total_frames = 0  # Counter to track the number of frames
        self.current_batch = []
        self.batch_size = 10000
        self.num = 1

    def add_node(self, node):
        self.process_path(path_states(node))

    def save_frame(self, path, frame_num):
        print("Processing frame", frame_num, end="\r")
        """Render a single frame and save it as an image."""
        fig, ax = plt.subplots()

        # Manually drawing the gridlines
        for x in range(self.grid_size[1] + 1):
            ax.plot([x, x], [0, self.grid_size[0]], color="black", linewidth=1)
        for y in range(self.grid_size[0] + 1):
            ax.plot([0, self.grid_size[1]], [y, y], color="black", linewidth=1)

        # Draw the black obstacles inside the grid
        for obstacle in self.blocks:
            rect = plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, facecolor="black")
            ax.add_patch(rect)

        # Mark the start and goal positions within the grid
        ax.text(
            self.initial[1] + 0.5,
            self.initial[0] + 0.5,
            "S",
            ha="center",
            va="center",
            color="violet",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            self.goal[1] + 0.5,
            self.goal[0] + 0.5,
            "G",
            ha="center",
            va="center",
            color="violet",
            fontsize=12,
            fontweight="bold",
        )

        # Plot the path with a gradient of colors
        colors = cm.rainbow(np.linspace(0, 1, len(path)))
        for i, (x, y) in enumerate(path):
            ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=colors[i]))
            ax.text(
                y + 0.5,
                x + 0.5,
                f"{i+1}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

        # Set labels for axes
        ax.set_xlabel("Y axis", fontsize=12, fontweight="bold")
        ax.set_ylabel("X axis", fontsize=12, fontweight="bold")

        # Set the ticks to align with the centers of squares
        ax.set_xticks(np.arange(0.5, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0.5, self.grid_size[0], 1))
        ax.set_xticklabels(np.arange(0, self.grid_size[1], 1))
        ax.set_yticklabels(np.arange(0, self.grid_size[0], 1))

        # Adjust the limits so that the ticks correspond to the little squares
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])

        # Save the current frame as an image
        frame_path = os.path.join(self.save_dir, f"{frame_num}.png")
        plt.savefig(frame_path)
        plt.close()

    def process_path(self, path):
        """Cache the path in memory and process the batch when needed."""
        self.current_batch.append(path)

        # If batch size limit reached, process and save all frames in the batch
        if len(self.current_batch) >= self.batch_size:
            # self.save_batch()
            self.process_path_multiprocessing()

    def process_path_multiprocessing(self):
        """Process paths using multiprocessing and save images."""
        # Use multiprocessing pool with number of workers equal to the number of CPU cores
        print(f"\nProcessing batch...{self.total_frames + len(self.current_batch)}")
        with Pool(cpu_count() - 2) as pool:
            # Distribute the work across processes
            pool.starmap(
                self.save_frame,
                [
                    (path, frame_num)
                    for frame_num, path in enumerate(self.current_batch, start=self.num)
                ],
            )
        self.total_frames += len(self.current_batch)
        self.num += len(self.current_batch)
        self.current_batch = []
        print(f"Batch processed {self.total_frames}")

    def finish_processing(self):
        """Call this method after processing all paths to save any remaining cached paths."""
        if self.current_batch:
            print("Saving remaining paths...")
            self.process_path_multiprocessing()
            # self.save_batch()

    def create_video(self, output_video="pathfinding.mp4"):
        """Create a video from the saved frames."""
        self.finish_processing()
        clip = ImageSequenceClip(
            [
                os.path.join(self.save_dir, f"{i}.png")
                for i in range(1, self.total_frames + 1)
            ],
            fps=24,
        )
        clip.write_videofile(output_video, codec="libx264")

        print(f"Video saved as {output_video}")
