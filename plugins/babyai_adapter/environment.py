"""
Hermetic, spec-compliant MiniGrid & BabyAI Environment simulator.

Provides a fast, zero-dependency implementation of the BabyAI Level-1 single-room
environment (GoToObj and PickupObj) adhering strictly to Gymnasium/MiniGrid conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .types import (
    DIR_TO_VEC,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    MiniGridAction,
    MiniGridColor,
    MiniGridDirection,
    MiniGridObjectType,
    MiniGridObservation,
    MiniGridState,
)


@dataclass
class GridCell:
    object_type: MiniGridObjectType = MiniGridObjectType.EMPTY
    color: MiniGridColor = MiniGridColor.RED
    state: MiniGridState = MiniGridState.OPEN

    def to_tuple(self) -> tuple[int, int, int]:
        return (int(self.object_type), int(self.color), int(self.state))


class BabyAIEnvironment:
    """Zero-dependency MiniGrid simulator for BabyAI single-room levels."""

    def __init__(
        self,
        room_size: int = 8,
        width: int | None = None,
        height: int | None = None,
        max_steps: int = 64,
        mission: str = "go to the red ball",
    ) -> None:
        self.width = width or room_size
        self.height = height or room_size
        self.max_steps = max_steps
        self.mission = mission
        self.step_count = 0

        self.agent_pos: tuple[int, int] = (1, 1)
        self.agent_dir: int = int(MiniGridDirection.EAST)
        self.carrying: GridCell | None = None

        # 2D Grid [x][y]
        self.grid: list[list[GridCell]] = [
            [GridCell() for _ in range(self.height)] for _ in range(self.width)
        ]
        self._init_outer_walls()

    def _init_outer_walls(self) -> None:
        """Surround room perimeter with wall cells."""
        for x in range(self.width):
            for y in range(self.height):
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.grid[x][y] = GridCell(
                        object_type=MiniGridObjectType.WALL,
                        color=MiniGridColor.GREY,
                        state=MiniGridState.CLOSED,
                    )
                else:
                    self.grid[x][y] = GridCell(
                        object_type=MiniGridObjectType.EMPTY,
                        color=MiniGridColor.RED,
                        state=MiniGridState.OPEN,
                    )

    def place_object(
        self,
        x: int,
        y: int,
        object_type: MiniGridObjectType,
        color: MiniGridColor,
        state: MiniGridState | None = None,
    ) -> None:
        """Place an object at discrete coordinates."""
        if state is None:
            state = (
                MiniGridState.CLOSED
                if object_type == MiniGridObjectType.DOOR
                else MiniGridState.OPEN
            )
        if 0 < x < self.width - 1 and 0 < y < self.height - 1:
            self.grid[x][y] = GridCell(object_type=object_type, color=color, state=state)

    def get_front_pos(self) -> tuple[int, int]:
        """Get coordinates directly in front of the agent."""
        dx, dy = DIR_TO_VEC[MiniGridDirection(self.agent_dir)]
        return (self.agent_pos[0] + dx, self.agent_pos[1] + dy)

    def gen_obs(self) -> MiniGridObservation:
        """Generate canonical 7x7x3 partial grid observation in front of agent."""
        view_size = 7
        obs_grid = [[[0, 0, 0] for _ in range(view_size)] for _ in range(view_size)]

        # In MiniGrid, agent is at bottom center (3, 6) facing "up" in the partial view
        fwd_vec = DIR_TO_VEC[MiniGridDirection(self.agent_dir)]
        # right_vec is 90 deg clockwise from fwd_vec
        right_vec = (-fwd_vec[1], fwd_vec[0])

        for vx in range(view_size):
            for vy in range(view_size):
                fwd_dist = 6 - vy
                right_dist = vx - 3

                wx = self.agent_pos[0] + fwd_dist * fwd_vec[0] + right_dist * right_vec[0]
                wy = self.agent_pos[1] + fwd_dist * fwd_vec[1] + right_dist * right_vec[1]

                if 0 <= wx < self.width and 0 <= wy < self.height:
                    # Line of sight check: is there a wall blocking between agent and (wx, wy)?
                    if self._is_line_of_sight_blocked(self.agent_pos, (wx, wy)):
                        obs_grid[vx][vy] = [int(MiniGridObjectType.UNSEEN), 0, 0]
                    else:
                        cell = self.grid[wx][wy]
                        obs_grid[vx][vy] = list(cell.to_tuple())
                else:
                    # Outside world boundary
                    obs_grid[vx][vy] = [int(MiniGridObjectType.UNSEEN), 0, 0]

        return MiniGridObservation(
            image=obs_grid,
            direction=self.agent_dir,
            mission=self.mission,
            step_count=self.step_count,
            extra={
                "agent_pos": self.agent_pos,
                "carrying": self.carrying.to_tuple() if self.carrying else None,
            },
        )

    def _is_line_of_sight_blocked(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        """Bresenham raycast line-of-sight obstruction check."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        curr_x, curr_y = x0, y0
        while (curr_x, curr_y) != (x1, y1):
            if (curr_x, curr_y) != (x0, y0):
                # If this intermediate cell blocks vision:
                if 0 <= curr_x < self.width and 0 <= curr_y < self.height:
                    cell = self.grid[curr_x][curr_y]
                    if cell.object_type in (MiniGridObjectType.WALL,):
                        return True
                    if cell.object_type == MiniGridObjectType.DOOR and cell.state in (
                        MiniGridState.CLOSED,
                        MiniGridState.LOCKED,
                    ):
                        return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                curr_x += sx
            if e2 < dx:
                err += dx
                curr_y += sy

        return False

    def reset(self) -> MiniGridObservation:
        """Reset step count and return initial observation."""
        self.step_count = 0
        return self.gen_obs()

    def step(
        self, action: int | MiniGridAction
    ) -> tuple[MiniGridObservation, float, bool, bool, dict[str, Any]]:
        """Execute discrete MiniGrid action."""
        self.step_count += 1
        act = MiniGridAction(action)
        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_steps

        fwd_vec = DIR_TO_VEC[MiniGridDirection(self.agent_dir)]

        if act == MiniGridAction.LEFT:
            self.agent_dir = (self.agent_dir - 1) % 4

        elif act == MiniGridAction.RIGHT:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif act == MiniGridAction.FORWARD:
            target_x = self.agent_pos[0] + fwd_vec[0]
            target_y = self.agent_pos[1] + fwd_vec[1]
            if 0 <= target_x < self.width and 0 <= target_y < self.height:
                cell = self.grid[target_x][target_y]
                # Passable if empty or floor or open door
                if cell.object_type in (MiniGridObjectType.EMPTY, MiniGridObjectType.FLOOR) or (
                    cell.object_type == MiniGridObjectType.DOOR and cell.state == MiniGridState.OPEN
                ):
                    self.agent_pos = (target_x, target_y)

        elif act == MiniGridAction.PICKUP:
            fx, fy = self.get_front_pos()
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fx][fy]
                if self.carrying is None and front_cell.object_type in (
                    MiniGridObjectType.BALL,
                    MiniGridObjectType.BOX,
                    MiniGridObjectType.KEY,
                ):
                    self.carrying = front_cell
                    self.grid[fx][fy] = GridCell(
                        object_type=MiniGridObjectType.EMPTY,
                        color=MiniGridColor.RED,
                        state=MiniGridState.OPEN,
                    )

        elif act == MiniGridAction.DROP:
            fx, fy = self.get_front_pos()
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fx][fy]
                if self.carrying is not None and front_cell.object_type in (
                    MiniGridObjectType.EMPTY,
                    MiniGridObjectType.FLOOR,
                ):
                    self.grid[fx][fy] = self.carrying
                    self.carrying = None

        elif act == MiniGridAction.TOGGLE:
            fx, fy = self.get_front_pos()
            if 0 <= fx < self.width and 0 <= fy < self.height:
                front_cell = self.grid[fx][fy]
                if front_cell.object_type == MiniGridObjectType.DOOR:
                    if front_cell.state == MiniGridState.LOCKED:
                        # Can only open if carrying matching key!
                        if (
                            self.carrying is not None
                            and self.carrying.object_type == MiniGridObjectType.KEY
                            and self.carrying.color == front_cell.color
                        ):
                            front_cell.state = MiniGridState.OPEN
                    elif front_cell.state == MiniGridState.OPEN:
                        front_cell.state = MiniGridState.CLOSED
                    else:  # CLOSED
                        front_cell.state = MiniGridState.OPEN

        elif act == MiniGridAction.DONE:
            # Check success condition
            if self._check_goal_achieved():
                terminated = True
                reward = max(0.0, 1.0 - 0.9 * (self.step_count / self.max_steps))

        # Check automatic goal termination for go_to
        if not terminated and self._check_goal_achieved():
            # In GoTo, if agent is adjacent to the target object and facing it
            terminated = True
            reward = max(0.0, 1.0 - 0.9 * (self.step_count / self.max_steps))

        obs = self.gen_obs()
        info = {
            "step_count": self.step_count,
            "agent_pos": self.agent_pos,
            "agent_dir": self.agent_dir,
            "carrying": self.carrying.to_tuple() if self.carrying else None,
            "goal_achieved": terminated and reward > 0.0,
        }
        return obs, reward, terminated, truncated, info

    def _is_single_goal_achieved(self, goal: Any) -> bool:
        """Evaluate if a single BabyAIGoal is satisfied on the environment."""
        if goal.action == "pickup":
            if self.carrying is None:
                return False
            car_type = IDX_TO_OBJECT.get(int(self.carrying.object_type), "")
            car_color = IDX_TO_COLOR.get(int(self.carrying.color), "")
            return goal.matches_attributes(car_type, car_color)

        if goal.action == "open":
            for x in range(self.width):
                for y in range(self.height):
                    cell = self.grid[x][y]
                    if cell.object_type == MiniGridObjectType.DOOR:
                        door_col = IDX_TO_COLOR.get(int(cell.color), "")
                        if cell.state == MiniGridState.OPEN:
                            if goal.target_color is None or goal.target_color == door_col:
                                return True
            return False

        if goal.action == "put_next":
            m_cells = []
            f_cells = []
            for x in range(self.width):
                for y in range(self.height):
                    cell = self.grid[x][y]
                    c_type = IDX_TO_OBJECT.get(int(cell.object_type), "")
                    c_col = IDX_TO_COLOR.get(int(cell.color), "")
                    if goal.matches_attributes(c_type, c_col):
                        m_cells.append((x, y))
                    if goal.matches_fixed_attributes(c_type, c_col):
                        f_cells.append((x, y))
            for mx, my in m_cells:
                for fx, fy in f_cells:
                    if abs(mx - fx) + abs(my - fy) == 1:
                        return True
            return False

        if goal.action == "go_to":
            fx, fy = self.get_front_pos()
            if 0 <= fx < self.width and 0 <= fy < self.height:
                cell = self.grid[fx][fy]
                c_type = IDX_TO_OBJECT.get(int(cell.object_type), "")
                c_col = IDX_TO_COLOR.get(int(cell.color), "")
                if goal.matches_attributes(c_type, c_col):
                    return True
            return False

        return False

    def _check_goal_achieved(self) -> bool:
        """Evaluate if the goal specified in the mission string is satisfied."""
        from .mission import BabyAIMissionParser

        parser = BabyAIMissionParser()
        parsed_goal = parser.parse(self.mission)

        if parsed_goal.is_compound():
            for sub in parsed_goal.subgoals:
                if not self._is_single_goal_achieved(sub):
                    return False
            return True

        return self._is_single_goal_achieved(parsed_goal)


def create_babyai_level(
    mission: str,
    target: tuple[str, str, tuple[int, int]],
    distractors: list[tuple[str, str, tuple[int, int]]],
    agent_pos: tuple[int, int] = (1, 1),
    agent_dir: int = 0,
    room_size: int = 8,
) -> BabyAIEnvironment:
    """Helper to instantiate a deterministic BabyAI room layout."""
    env = BabyAIEnvironment(room_size=room_size, mission=mission)
    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

    # Place target
    tgt_type_str, tgt_col_str, (tx, ty) = target
    tgt_type = MiniGridObjectType[tgt_type_str.upper()]
    tgt_col = MiniGridColor[tgt_col_str.upper()]
    env.place_object(tx, ty, tgt_type, tgt_col)

    # Place distractors
    for d_type_str, d_col_str, (dx, dy) in distractors:
        d_type = MiniGridObjectType[d_type_str.upper()]
        d_col = MiniGridColor[d_col_str.upper()]
        env.place_object(dx, dy, d_type, d_col)

    return env


def create_two_room_door_level(
    mission: str = "open the red door",
    door_color: str = "red",
    door_pos: tuple[int, int] = (4, 2),
    door_state: MiniGridState = MiniGridState.CLOSED,
    room_width: int = 9,
    room_height: int = 5,
    agent_pos: tuple[int, int] = (2, 2),
    agent_dir: int = 0,
    target_in_room2: tuple[str, str, tuple[int, int]] | None = None,
    distractors: list[tuple[str, str, tuple[int, int]]] | None = None,
) -> BabyAIEnvironment:
    """Instantiate a two-room partitioned environment (similar to BabyAI-OpenRedDoor-v0)."""
    env = BabyAIEnvironment(width=room_width, height=room_height, mission=mission)
    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

    split_x = door_pos[0]
    # Build partition wall at split_x
    for y in range(room_height):
        if y == door_pos[1]:
            d_col = MiniGridColor[door_color.upper()]
            env.grid[split_x][y] = GridCell(
                object_type=MiniGridObjectType.DOOR,
                color=d_col,
                state=door_state,
            )
        else:
            env.grid[split_x][y] = GridCell(
                object_type=MiniGridObjectType.WALL,
                color=MiniGridColor.GREY,
                state=MiniGridState.CLOSED,
            )

    if target_in_room2:
        t_type, t_col, (tx, ty) = target_in_room2
        env.place_object(tx, ty, MiniGridObjectType[t_type.upper()], MiniGridColor[t_col.upper()])

    if distractors:
        for d_type, d_col, (dx, dy) in distractors:
            env.place_object(
                dx, dy, MiniGridObjectType[d_type.upper()], MiniGridColor[d_col.upper()]
            )

    return env


def create_unlock_door_level(
    mission: str = "open the red door",
    door_color: str = "red",
    door_pos: tuple[int, int] = (4, 2),
    key_pos: tuple[int, int] = (2, 2),
    key_color: str | None = None,
    distractors: list[tuple[str, str, tuple[int, int]]] | None = None,
    agent_pos: tuple[int, int] = (1, 1),
    agent_dir: int = 0,
    room_width: int = 9,
    room_height: int = 5,
) -> BabyAIEnvironment:
    """Instantiate a partitioned level with a locked door and a key."""
    env = BabyAIEnvironment(width=room_width, height=room_height, mission=mission)
    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

    split_x = door_pos[0]
    # Build partition wall at split_x with locked door
    for y in range(room_height):
        if y == door_pos[1]:
            d_col = MiniGridColor[door_color.upper()]
            env.grid[split_x][y] = GridCell(
                object_type=MiniGridObjectType.DOOR,
                color=d_col,
                state=MiniGridState.LOCKED,
            )
        else:
            env.grid[split_x][y] = GridCell(
                object_type=MiniGridObjectType.WALL,
                color=MiniGridColor.GREY,
                state=MiniGridState.CLOSED,
            )

    # Place key (matching door color by default)
    kc = key_color or door_color
    env.place_object(
        key_pos[0],
        key_pos[1],
        MiniGridObjectType.KEY,
        MiniGridColor[kc.upper()],
        state=MiniGridState.OPEN,
    )

    if distractors:
        for d_type, d_col, (dx, dy) in distractors:
            env.place_object(
                dx, dy, MiniGridObjectType[d_type.upper()], MiniGridColor[d_col.upper()]
            )

    return env


def create_put_next_level(
    mission: str = "put the yellow key next to the yellow box",
    move_obj: tuple[str, str, tuple[int, int]] = ("key", "yellow", (2, 2)),
    fixed_obj: tuple[str, str, tuple[int, int]] = ("box", "yellow", (5, 4)),
    distractors: list[tuple[str, str, tuple[int, int]]] | None = None,
    agent_pos: tuple[int, int] = (1, 1),
    agent_dir: int = 0,
    room_size: int = 8,
) -> BabyAIEnvironment:
    """Instantiate a single room level for relational PutNext tasks."""
    env = BabyAIEnvironment(width=room_size, height=room_size, mission=mission)
    env.agent_pos = agent_pos
    env.agent_dir = agent_dir

    m_type, m_col, (mx, my) = move_obj
    env.place_object(mx, my, MiniGridObjectType[m_type.upper()], MiniGridColor[m_col.upper()])

    f_type, f_col, (fx, fy) = fixed_obj
    env.place_object(fx, fy, MiniGridObjectType[f_type.upper()], MiniGridColor[f_col.upper()])

    if distractors:
        for d_type, d_col, (dx, dy) in distractors:
            env.place_object(
                dx, dy, MiniGridObjectType[d_type.upper()], MiniGridColor[d_col.upper()]
            )

    return env


def create_blocked_level(
    mission: str = "pick up the red box",
    target_pos: tuple[int, int] = (7, 2),
    door_pos: tuple[int, int] = (4, 2),
    blocker_pos: tuple[int, int] = (3, 2),
    blocker: tuple[str, str] = ("ball", "purple"),
    key_pos: tuple[int, int] = (2, 1),
    agent_pos: tuple[int, int] = (1, 1),
    room_width: int = 9,
    room_height: int = 5,
) -> BabyAIEnvironment:
    """Instantiate a partitioned level where the door is blocked by a pickupable obstacle."""
    env = create_unlock_door_level(
        mission=mission,
        door_color="red",
        door_pos=door_pos,
        key_pos=key_pos,
        agent_pos=agent_pos,
        room_width=room_width,
        room_height=room_height,
    )
    b_type, b_col = blocker
    env.place_object(
        blocker_pos[0],
        blocker_pos[1],
        MiniGridObjectType[b_type.upper()],
        MiniGridColor[b_col.upper()],
    )
    env.place_object(
        target_pos[0],
        target_pos[1],
        MiniGridObjectType.BOX,
        MiniGridColor.RED,
    )
    return env


def create_sequential_level(
    mission: str = "open the red door, then pick up the green box",
    door_pos: tuple[int, int] = (4, 2),
    target_pos: tuple[int, int] = (7, 2),
    agent_pos: tuple[int, int] = (1, 1),
    room_width: int = 9,
    room_height: int = 5,
) -> BabyAIEnvironment:
    """Instantiate a level with sequential subgoals across rooms."""
    return create_two_room_door_level(
        mission=mission,
        door_color="red",
        door_pos=door_pos,
        door_state=MiniGridState.CLOSED,
        target_in_room2=("box", "green", target_pos),
        agent_pos=agent_pos,
        room_width=room_width,
        room_height=room_height,
    )


def make_gym_babyai_level(
    env_id: str = "BabyAI-GoToObj-v0",
    render_mode: str | None = None,
    **kwargs: Any,
) -> Any:
    """Instantiate the upstream official Farama Gymnasium BabyAI environment.

    Requires 'minigrid' and 'gymnasium'.
    """
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401

        return gym.make(env_id, render_mode=render_mode, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"Loading official Gym environment '{env_id}' requires 'minigrid' and 'gymnasium'. "
            f"Install them via: pip install minigrid gymnasium"
        ) from e
