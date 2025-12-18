from __future__ import annotations

from minigrid.core.world_object import WorldObj, Ball
from minigrid.utils.rendering import fill_coords, point_in_rect
from minigrid.core.actions import Actions
from minigrid.core.constants import COLOR_NAMES, COLORS, OBJECT_TO_IDX, COLOR_TO_IDX

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
        box_color = COLORS[self.color]

        # Main box body
        fill_coords(img, point_in_rect(0.10, 0.90, 0.10, 0.90), box_color)

        # Center window: shows empty/full
        if self.contains is None:
            inner_color = (0, 0, 0)  # empty = black
        else:
            inner_color = COLORS[self.contains.color]  # full = contained object's color

        # Central square window
        fill_coords(img, point_in_rect(0.30, 0.70, 0.35, 0.75), inner_color)

        # Open/closed indicator band
        # Use a light band color so it contrasts with box + window
        band_color = (255, 255, 255)

        if self.is_open:
            # OPEN: vertical band in the center
            fill_coords(
                img,
                point_in_rect(0.47, 0.53, 0.10, 0.90),
                band_color,
            )
        else:
            # CLOSED: diagonal band (top-left -> bottom-right)
            def diag_band(x, y):
                # Thin diagonal strip where y ≈ x + offset
                return (y > x + 0.05) and (y < x + 0.15)

            fill_coords(img, diag_band, band_color)

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
        agent_start_pos=(1, 1),
        agent_start_dir: int = 0,
        num_boxes: int = 3,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        self.num_boxes = num_boxes
        self.box_colors = []  # filled in _gen_grid
        self.boxes: list[ContainerBox] = []

        mission_space = MissionSpace(mission_func=self._gen_mission)
        self.step_count = 0
        self.prev_correct = 0

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
        for c in self.box_colors:
            self.place_obj(Ball(color=c))

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
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
        # Everything else: default MiniGrid behavior
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1
        return obs, reward, terminated, truncated, info
