#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maze shortest-path planner based on a reinforcement-learning Bellman/Q update.

Usage:
  python maze_rl_gui.py maze.jpg
  python maze_rl_gui.py maze.jpg --target 25 33 --save solved.png
  python maze_rl_gui.py maze.jpg --pixel 558 404 --save solved.png

Dependencies:
  pip install numpy pillow matplotlib
"""
from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parent.parent / ".matplotlib_cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
NEG = -1_000_000.0
RANDOM_ROUTE_SEED = 7
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Q7_outputs"
PATH_COLOR = "#c44e52"
START_COLOR = "#e0b84d"
PATH_HALO_COLOR = "white"


def configure_chinese_font() -> None:
    """Use an installed CJK font so Matplotlib GUI text renders Chinese correctly."""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class PlanResult:
    ok: bool
    message: str
    path: list[tuple[int, int]]
    q_iterations: int = 0


class RLMazeSolver:
    """Parse the attached maze image and solve arbitrary goals with Q-iteration."""

    def __init__(self, image_path: str | Path, cell_size: Optional[int] = None) -> None:
        self.image_path = Path(image_path)
        self.rgb = np.asarray(Image.open(self.image_path).convert("RGB"))
        self.gray = np.asarray(Image.open(self.image_path).convert("L"))
        self.height_px, self.width_px = self.gray.shape

        self.start_px = self._detect_start_pixel()
        self.x0, self.y0, self.x1, self.y1, auto_cell = self._detect_maze_box()
        self.cell = int(cell_size or auto_cell)
        self.rows = (self.y1 - self.y0 + 1) // self.cell
        self.cols = (self.x1 - self.x0 + 1) // self.cell

        self.grid = self._image_to_grid()  # True = road, False = wall
        self.start = self.pixel_to_cell(*self.start_px)
        if self.start is None:
            raise ValueError("Detected start point is outside the maze box.")
        self.grid[self.start] = True  # the yellow marker covers a road block, so force it open

    def _detect_start_pixel(self) -> tuple[int, int]:
        r = self.rgb[:, :, 0].astype(int)
        g = self.rgb[:, :, 1].astype(int)
        b = self.rgb[:, :, 2].astype(int)
        yellow_score = r + g - 2 * b
        yellow = (r > 80) & (g > 70) & (yellow_score > 40) & (np.abs(r - g) < 80)
        ys, xs = np.where(yellow)
        if len(xs) == 0:
            raise ValueError("No yellow start marker was found in the image.")
        return int(round(xs.mean())), int(round(ys.mean()))

    def _detect_maze_box(self) -> tuple[int, int, int, int, int]:
        # The wall blocks are bright hatched blocks. A threshold of 40 isolates the rectangle well.
        bright = self.gray > 40
        ys, xs = np.where(bright)
        if len(xs) == 0:
            raise ValueError("Maze wall pixels were not found.")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        cell = math.gcd(w, h)
        if cell < 4:
            # Fallback for a slightly noisy image. The supplied maze is exactly 8 px per block.
            cell = 8
            w = (w // cell) * cell
            h = (h // cell) * cell
            x1, y1 = x0 + w - 1, y0 + h - 1
        return x0, y0, x1, y1, cell

    def _image_to_grid(self) -> np.ndarray:
        means = np.zeros((self.rows, self.cols), dtype=float)
        for row in range(self.rows):
            for col in range(self.cols):
                y0 = self.y0 + row * self.cell
                x0 = self.x0 + col * self.cell
                block = self.gray[y0 : y0 + self.cell, x0 : x0 + self.cell]
                means[row, col] = block.mean()

        values = np.sort(means.ravel())
        gaps = np.diff(values)
        threshold = (values[int(np.argmax(gaps))] + values[int(np.argmax(gaps)) + 1]) / 2.0
        return means < threshold

    def pixel_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return None
        col = int((x - self.x0) // self.cell)
        row = int((y - self.y0) // self.cell)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.x0 + (col + 0.5) * self.cell,
            self.y0 + (row + 0.5) * self.cell,
        )

    def _valid_neighbors(self, row: int, col: int) -> Iterable[tuple[int, int]]:
        for dr, dc in DIRS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr, nc]:
                yield nr, nc

    def _reachable_from_start(self, goal: tuple[int, int]) -> bool:
        if not self.grid[goal]:
            return False
        q: deque[tuple[int, int]] = deque([self.start])
        visited = {self.start}
        while q:
            state = q.popleft()
            if state == goal:
                return True
            for nxt in self._valid_neighbors(*state):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return False

    def reachable_cells_from_start(self) -> list[tuple[int, int]]:
        q: deque[tuple[int, int]] = deque([self.start])
        visited = {self.start}
        while q:
            state = q.popleft()
            for nxt in self._valid_neighbors(*state):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return sorted(visited)

    def q_iteration(self, goal: tuple[int, int], max_iter: Optional[int] = None) -> tuple[np.ndarray, int, bool]:
        """
        Bellman optimality update on Q/V values.

        Reward design:
          - every non-terminal move gives -1
          - the goal is terminal with value 0
          - gamma = 1

        Therefore V*(s) = - shortest_distance(s, goal), and the greedy policy is a shortest path.
        """
        max_iter = max_iter or (self.rows * self.cols + 5)
        value = np.full((self.rows, self.cols), NEG, dtype=float)
        value[goal] = 0.0

        for iteration in range(1, max_iter + 1):
            old = value

            up = np.full_like(old, NEG)
            down = np.full_like(old, NEG)
            left = np.full_like(old, NEG)
            right = np.full_like(old, NEG)

            up[1:, :] = -1 + old[:-1, :]
            up[1:, :][~self.grid[:-1, :]] = NEG

            down[:-1, :] = -1 + old[1:, :]
            down[:-1, :][~self.grid[1:, :]] = NEG

            left[:, 1:] = -1 + old[:, :-1]
            left[:, 1:][~self.grid[:, :-1]] = NEG

            right[:, :-1] = -1 + old[:, 1:]
            right[:, :-1][~self.grid[:, 1:]] = NEG

            new_value = np.maximum.reduce([up, down, left, right])
            new_value[~self.grid] = NEG
            new_value[goal] = 0.0

            if np.array_equal(new_value, value):
                return new_value, iteration, True
            value = new_value

        return value, max_iter, False

    def plan_to_cell(self, goal: tuple[int, int]) -> PlanResult:
        row, col = goal
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return PlanResult(False, f"目标格 {goal} 超出迷宫范围。", [])
        if not self.grid[row, col]:
            return PlanResult(False, f"目标格 {goal} 是墙，不能到达。", [])
        if not self._reachable_from_start(goal):
            return PlanResult(False, f"目标格 {goal} 与起点不连通，不能到达。", [])

        value, iters, converged = self.q_iteration(goal)
        if value[self.start] <= NEG / 2:
            return PlanResult(False, f"目标格 {goal} 与起点不连通，不能到达。", [], iters)

        path = [self.start]
        current = self.start
        seen = {current}

        while current != goal:
            candidates = list(self._valid_neighbors(*current))
            if not candidates:
                return PlanResult(False, "路径提取失败：当前位置没有可走邻居。", [], iters)
            nxt = max(candidates, key=lambda p: value[p])
            if value[nxt] <= NEG / 2 or nxt in seen:
                return PlanResult(False, "路径提取失败：Q 值没有形成可达策略。", [], iters)
            path.append(nxt)
            seen.add(nxt)
            current = nxt

        suffix = "" if converged else "；Q 迭代达到上限，结果需检查"
        return PlanResult(True, f"目标格 {goal} 可达，最短路径长度 {len(path) - 1} 步，Q 迭代 {iters} 轮{suffix}。", path, iters)

    def plan_to_pixel(self, x: float, y: float) -> PlanResult:
        goal = self.pixel_to_cell(x, y)
        if goal is None:
            return PlanResult(False, "点击位置不在迷宫内部。", [])
        return self.plan_to_cell(goal)

    def draw_solution(self, path: list[tuple[int, int]], save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        configure_chinese_font()
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(self.rgb)
        ax.axis("off")
        xs, ys = zip(*(self.cell_center(r, c) for r, c in path))
        ax.plot(xs, ys, color=PATH_HALO_COLOR, linewidth=4.0, solid_capstyle="round")
        ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.2, solid_capstyle="round")
        ax.scatter([xs[0]], [ys[0]], s=46, c=START_COLOR, edgecolors="black", linewidths=0.8, label="start")
        ax.scatter([xs[-1]], [ys[-1]], s=72, c=PATH_COLOR, edgecolors="black", linewidths=0.8, marker="X", label="goal")
        ax.legend(loc="lower left")
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)


def resolve_output_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    if path.is_absolute():
        return path
    return OUTPUT_DIR / path


def save_random_startup_routes(solver: RLMazeSolver, count: int = 2, seed: int = RANDOM_ROUTE_SEED) -> None:
    candidates = [cell for cell in solver.reachable_cells_from_start() if cell != solver.start]
    if len(candidates) < count:
        print(f"random routes skipped: only {len(candidates)} reachable non-start road cells found")
        return

    rng = random.Random(seed)
    goals = rng.sample(candidates, count)
    print(f"random route seed: {seed}")
    for index, goal in enumerate(goals, start=1):
        result = solver.plan_to_cell(goal)
        print(f"random goal {index}: {goal}; {result.message}")
        if result.ok:
            save_path = OUTPUT_DIR / f"random_route_{index}.png"
            solver.draw_solution(result.path, save_path)
            print(f"saved random route {index}: {save_path}")


def run_gui(solver: RLMazeSolver) -> None:
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(solver.rgb)
    ax.set_title("单击任意目标点；红线为 RL/Bellman Q 迭代规划出的最短路径；按 Esc 退出")
    ax.axis("off")

    message = ax.text(
        0.02,
        0.03,
        f"起点像素 {solver.start_px}，起点格 {solver.start}。",
        transform=ax.transAxes,
        fontsize=10,
        color=START_COLOR,
        bbox=dict(facecolor="black", alpha=0.65, pad=4),
    )
    artists = []

    def clear_artists() -> None:
        while artists:
            artist = artists.pop()
            try:
                artist.remove()
            except ValueError:
                pass

    def onclick(event) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        clear_artists()
        result = solver.plan_to_pixel(event.xdata, event.ydata)
        message.set_text(result.message)
        if result.ok:
            xs, ys = zip(*(solver.cell_center(r, c) for r, c in result.path))
            (halo,) = ax.plot(xs, ys, color=PATH_HALO_COLOR, linewidth=4.0, solid_capstyle="round")
            (line,) = ax.plot(xs, ys, color=PATH_COLOR, linewidth=2.2, solid_capstyle="round")
            goal_marker = ax.scatter(
                [xs[-1]],
                [ys[-1]],
                s=80,
                c=PATH_COLOR,
                edgecolors="black",
                linewidths=0.8,
                marker="X",
            )
            artists.extend([halo, line, goal_marker])
        fig.canvas.draw_idle()

    def onkey(event) -> None:
        if event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default=SCRIPT_DIR / "maze.jpg", help="maze image path")
    parser.add_argument("--target", nargs=2, type=int, metavar=("ROW", "COL"), help="target logical grid cell")
    parser.add_argument("--pixel", nargs=2, type=float, metavar=("X", "Y"), help="target pixel position in the image")
    parser.add_argument("--cell", type=int, default=None, help="override detected logical cell size")
    parser.add_argument("--save", default="maze_rl_path.png", help="output image path for non-GUI mode")
    args = parser.parse_args()
    save_path = resolve_output_path(args.save)

    solver = RLMazeSolver(args.image, cell_size=args.cell)
    print(f"maze grid: {solver.rows} rows x {solver.cols} cols; cell={solver.cell}px")
    print(f"start: pixel={solver.start_px}, cell={solver.start}")
    save_random_startup_routes(solver)

    if args.target is not None:
        result = solver.plan_to_cell((args.target[0], args.target[1]))
        print(result.message)
        if result.ok:
            solver.draw_solution(result.path, save_path)
            print(f"saved: {save_path}")
    elif args.pixel is not None:
        result = solver.plan_to_pixel(args.pixel[0], args.pixel[1])
        print(result.message)
        if result.ok:
            solver.draw_solution(result.path, save_path)
            print(f"saved: {save_path}")
    else:
        run_gui(solver)


if __name__ == "__main__":
    main()
