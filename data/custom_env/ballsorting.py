from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj, Ball
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle
from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, COLORS, OBJECT_TO_IDX, COLOR_TO_IDX
from typing import Optional
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

class ContainerBox(WorldObj):
    """
    Permanent container:
    - not movable
    - not destroyed on toggle
    - can store exactly 1 object (self.contains)
    - has open/close state (self.is_open)
    """
    def __init__(self, color: str, contains: WorldObj | None = None, is_open: bool = True):
        super().__init__("box", color)
        self.contains = contains
        self.is_open = is_open

    def can_pickup(self):
        # Permanent / fixed
        return False

    def can_overlap(self):
        # Occupies the cell like a normal object
        return False

    def toggle(self, env, pos):
        # Open/close without replacing self on the grid
        self.is_open = not self.is_open
        return True

    def encode(self):
        """
        Keep it MiniGrid-convention: (OBJECT_IDX, COLOR_IDX, STATE)
        STATE encodes open/closed + empty/full:
          0: open empty
          1: open full
          2: closed empty
          3: closed full
        """
        state = (0 if self.is_open else 2) + (0 if self.contains is None else 1)
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        box_col = COLORS[self.color]
        # Decide what "content color" is (what we want visible even when closed)
        if self.contains is None:
            content_col = (0, 0, 0)  # empty = black (or pick a gray)
        else:
            content_col = COLORS[self.contains.color]

        # --- Base: box body (always) ---
        # Outer body
        fill_coords(img, point_in_rect(0.05, 0.95, 0.05, 0.95), box_col)

        # Optional: inner cavity background for contrast (helps readability)
        # fill_coords(img, point_in_rect(0.20, 0.80, 0.20, 0.80), (0,0,0))

        if self.is_open:
            # =========================
            # OPEN: show "ball" inside
            # =========================
            # draw border around ball in container
            fill_coords(img, point_in_rect(0.15, 0.85, 0.15, 0.85), (0,0,0))

            # Draw the contained object as a circle (big and obvious)
            fill_coords(img, point_in_circle(0.50, 0.50, 0.30), content_col)

        else:
            # ==========================================
            # CLOSED: thick vertical band shows contents
            # plus colored side rails show box identity
            # ==========================================

            # create thick horizontal band upper and lower around center
            fill_coords(img, point_in_rect(0.05, 0.95, 0.05, 0.30), box_col)
            fill_coords(img, point_in_rect(0.05, 0.95, 0.70, 0.95), box_col)

            # create thick horizontal band in CONTENT color (always visible)
            fill_coords(img, point_in_rect(0.05, 0.95, 0.30, 0.70), content_col)

class BallSortingEnv(MiniGridEnv):
    """
    Empty room with colored boxes and balls.

    Goal: put each ball into the box of matching color.

    Boxes:
    - fixed at random positions
    - permanent ContainerBox objects (open/close, store 1 item)
    - initially open and empty

    Balls:
    - placed at random positions, not in boxes

    Reward:
    - Tiered: each new correctly sorted ball gives 1 / num_boxes
      * after 1 correct box: total 0.33 (if num_boxes=3)
      * after 2 correct boxes: total 0.67
      * after 3 correct boxes: total 1.0
    """

    def __init__(
        self,
        size: int = 8,
        num_boxes: int = 3,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.num_boxes = num_boxes
        self.box_colors = []  # filled in _gen_grid
        self.boxes: list[ContainerBox] = []
        self.balls: list[Ball] = []

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.step_count = 0
        self.prev_correct = 0
        self.carrying = None

        if max_steps is None:
            max_steps = (4 * size**2) * num_boxes

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs,
        )


    @staticmethod
    def _gen_mission():
        return "put balls into boxes with corresponding colors"

    # ---------- Grid generation ----------

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Sample colors per episode
        self.box_colors = self._rand_subset(COLOR_NAMES, self.num_boxes)

        # Place boxes: fixed random locations, open and empty
        self.boxes = []
        for c in self.box_colors:
            box = ContainerBox(color=c, contains=None, is_open=True)
            self.place_obj(box)
            self.boxes.append(box)

        # Place one ball per box color (not in boxes)
        self.balls = []
        for c in self.box_colors:
            ball = Ball(color=c)
            self.balls.append(ball)
            self.place_obj(ball)

        # Place the agent
        self.place_agent()

        self.mission = "put balls into boxes with corresponding colors"
        # track previous correct balls to only reward progress
        self.prev_correct = 0

    # ---------- Helper: correctness ----------

    def _count_correct(self) -> int:
        """Number of boxes containing a matching-color ball."""
        correct = 0
        for b in self.boxes:
            if (
                b.contains is not None
                and b.contains.type == "ball"
                and b.contains.color == b.color
            ):
                correct += 1
        return correct

    # ---------- Step with container logic + tiered reward ----------

    def step(self, action):
        fwd_pos = self.front_pos
        fwd_obj = self.grid.get(*fwd_pos)
        correct = self._count_correct()
        info = {}
        
        # TOGGLE: open/close box (no replacement)
        if action == Actions.toggle and isinstance(fwd_obj, ContainerBox):
            fwd_obj.toggle(self, fwd_pos)
            reward = 0.0
            info = {"event": "toggle_box", "num_correct": correct}

        # DROP into open, empty box
        elif (
            action == Actions.drop
            and self.carrying is not None
            and isinstance(fwd_obj, ContainerBox)
        ):
            if fwd_obj.is_open and fwd_obj.contains is None:
                fwd_obj.contains, self.carrying = self.carrying, None
                correct = self._count_correct()
                info = {"event": "drop_into_box", "num_correct": correct}

        # PICKUP from open box (take its contents into hand)
        elif (
            action == Actions.pickup
            and self.carrying is None
            and isinstance(fwd_obj, ContainerBox)
        ):
            if fwd_obj.is_open and fwd_obj.contains is not None:
                self.carrying, fwd_obj.contains = fwd_obj.contains, None
                info = {"event": "pickup_from_box", "num_correct":correct}
        # all other actions: default MiniGrid behavior
        else:
            obs, reward, terminated, truncated, info = super().step(action)
        
        # Perform generic operations for observation
        obs = self.gen_obs()
        truncated = self.step_count >= self.max_steps
        terminated = (correct == self.num_boxes)
        # update reward only proportional to new boxes that are solved
        if correct > self.prev_correct:
            reward = correct/self.num_boxes
            self.prev_correct = correct
        else:
            reward = 0.0
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info
