import numpy as np
import gymnasium
import json
import struct
import os
import pufferlib
from pufferlib.ocean.drive import binding


class Drive(pufferlib.PufferEnv):
    def __init__(
        self,
        render_mode=None,
        report_interval=1,
        width=1280,
        height=1024,
        human_agent_idx=0,
        reward_vehicle_collision=-0.1,
        reward_offroad_collision=-0.1,
        reward_goal=1.0,
        reward_goal_post_respawn=0.5,
        reward_ade=0.0,
        goal_radius=2.0,
        scenario_length=None,
        resample_frequency=91,
        num_maps=100,
        num_agents=512,
        action_type="discrete",
        control_all_agents=False,
        num_policy_controlled_agents=-1,
        deterministic_agent_selection=False,
        use_goal_generation=False,
        control_non_vehicles=False,
        buf=None,
        seed=1,
        init_steps=0,
    ):
        # env
        self.render_mode = render_mode
        self.num_maps = num_maps
        self.report_interval = report_interval
        self.reward_vehicle_collision = reward_vehicle_collision
        self.reward_offroad_collision = reward_offroad_collision
        self.reward_goal = reward_goal
        self.reward_goal_post_respawn = reward_goal_post_respawn
        self.goal_radius = goal_radius
        self.reward_ade = reward_ade
        self.human_agent_idx = human_agent_idx
        self.scenario_length = scenario_length
        self.control_non_vehicles = control_non_vehicles
        self.use_goal_generation = use_goal_generation
        self.resample_frequency = resample_frequency
        self.num_obs = 7 + 63 * 7 + 200 * 7
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(self.num_obs,), dtype=np.float32)
        self.init_steps = init_steps

        if action_type == "discrete":
            self.single_action_space = gymnasium.spaces.MultiDiscrete([7, 13])
        elif action_type == "continuous":
            self.single_action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"action_space must be 'discrete' or 'continuous'. Got: {action_type}")

        self._action_type_flag = 0 if action_type == "discrete" else 1

        # Check if resources directory exists
        binary_path = "resources/drive/binaries/map_000.bin"
        if not os.path.exists(binary_path):
            raise FileNotFoundError(
                f"Required directory {binary_path} not found. Please ensure the Drive maps are downloaded and installed correctly per docs."
            )

        # Check maps availability
        available_maps = len([name for name in os.listdir("resources/drive/binaries") if name.endswith(".bin")])
        if num_maps > available_maps:
            raise ValueError(
                f"num_maps ({num_maps}) exceeds available maps in directory ({available_maps}). Please reduce num_maps or add more maps to resources/drive/binaries."
            )
        self.control_all_agents = bool(control_all_agents)
        self.num_policy_controlled_agents = int(num_policy_controlled_agents)
        self.deterministic_agent_selection = bool(deterministic_agent_selection)

        agent_offsets, map_ids, num_envs = binding.shared(
            num_agents=num_agents,
            num_maps=num_maps,
            num_policy_controlled_agents=self.num_policy_controlled_agents,
            control_all_agents=1 if self.control_all_agents else 0,
            deterministic_agent_selection=1 if self.deterministic_agent_selection else 0,
        )
        self.num_agents = num_agents
        self.agent_offsets = agent_offsets
        self.map_ids = map_ids
        self.num_envs = num_envs
        super().__init__(buf=buf)
        env_ids = []
        for i in range(num_envs):
            cur = agent_offsets[i]
            nxt = agent_offsets[i + 1]
            env_id = binding.env_init(
                self.observations[cur:nxt],
                self.actions[cur:nxt],
                self.rewards[cur:nxt],
                self.terminals[cur:nxt],
                self.truncations[cur:nxt],
                seed,
                action_type=self._action_type_flag,
                human_agent_idx=human_agent_idx,
                reward_vehicle_collision=reward_vehicle_collision,
                reward_offroad_collision=reward_offroad_collision,
                reward_goal=reward_goal,
                reward_goal_post_respawn=reward_goal_post_respawn,
                reward_ade=reward_ade,
                goal_radius=goal_radius,
                scenario_length=(int(scenario_length) if scenario_length is not None else None),
                control_all_agents=1 if self.control_all_agents else 0,
                num_policy_controlled_agents=self.num_policy_controlled_agents,
                deterministic_agent_selection=1 if self.deterministic_agent_selection else 0,
                map_id=map_ids[i],
                max_agents=nxt - cur,
                ini_file="pufferlib/config/ocean/drive.ini",
                control_non_vehicles=int(control_non_vehicles),
                init_steps=init_steps,
            )
            env_ids.append(env_id)

        self.c_envs = binding.vectorize(*env_ids)

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.terminals[:] = 0
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1
        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
                # print(log)
        if self.tick > 0 and self.resample_frequency > 0 and self.tick % self.resample_frequency == 0:
            self.tick = 0
            will_resample = 1
            if will_resample:
                binding.vec_close(self.c_envs)
                agent_offsets, map_ids, num_envs = binding.shared(
                    num_agents=self.num_agents,
                    num_maps=self.num_maps,
                    num_policy_controlled_agents=self.num_policy_controlled_agents,
                    control_all_agents=1 if self.control_all_agents else 0,
                    deterministic_agent_selection=1 if self.deterministic_agent_selection else 0,
                )
                env_ids = []
                seed = np.random.randint(0, 2**32 - 1)
                for i in range(num_envs):
                    cur = agent_offsets[i]
                    nxt = agent_offsets[i + 1]
                    env_id = binding.env_init(
                        self.observations[cur:nxt],
                        self.actions[cur:nxt],
                        self.rewards[cur:nxt],
                        self.terminals[cur:nxt],
                        self.truncations[cur:nxt],
                        seed,
                        action_type=self._action_type_flag,
                        human_agent_idx=self.human_agent_idx,
                        reward_vehicle_collision=self.reward_vehicle_collision,
                        reward_offroad_collision=self.reward_offroad_collision,
                        reward_goal=self.reward_goal,
                        reward_goal_post_respawn=self.reward_goal_post_respawn,
                        reward_ade=self.reward_ade,
                        goal_radius=self.goal_radius,
                        scenario_length=(int(self.scenario_length) if self.scenario_length is not None else None),
                        control_all_agents=1 if self.control_all_agents else 0,
                        num_policy_controlled_agents=self.num_policy_controlled_agents,
                        deterministic_agent_selection=1 if self.deterministic_agent_selection else 0,
                        map_id=map_ids[i],
                        max_agents=nxt - cur,
                        ini_file="pufferlib/config/ocean/drive.ini",
                        control_non_vehicles=int(self.control_non_vehicles),
                        init_steps=self.init_steps,
                    )
                    env_ids.append(env_id)
                self.c_envs = binding.vectorize(*env_ids)

                binding.vec_reset(self.c_envs, seed)
                self.terminals[:] = 1
        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def get_state(self):
        try:
            return binding.vec_get(self.c_envs)
        except Exception:
            return binding.env_get(self.c_envs)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def calculate_area(p1, p2, p3):
    # Calculate the area of the triangle using the determinant method
    return 0.5 * abs((p1["x"] - p3["x"]) * (p2["y"] - p1["y"]) - (p1["x"] - p2["x"]) * (p3["y"] - p1["y"]))


def simplify_polyline(geometry, polyline_reduction_threshold):
    """Simplify the given polyline using a method inspired by Visvalingham-Whyatt, optimized for Python."""
    num_points = len(geometry)
    if num_points < 3:
        return geometry  # Not enough points to simplify

    skip = [False] * num_points
    skip_changed = True

    while skip_changed:
        skip_changed = False
        k = 0
        while k < num_points - 1:
            k_1 = k + 1
            while k_1 < num_points - 1 and skip[k_1]:
                k_1 += 1
            if k_1 >= num_points - 1:
                break

            k_2 = k_1 + 1
            while k_2 < num_points and skip[k_2]:
                k_2 += 1
            if k_2 >= num_points:
                break

            point1 = geometry[k]
            point2 = geometry[k_1]
            point3 = geometry[k_2]
            area = calculate_area(point1, point2, point3)

            if area < polyline_reduction_threshold:
                skip[k_1] = True
                skip_changed = True
                k = k_2
            else:
                k = k_1

    return [geometry[i] for i in range(num_points) if not skip[i]]


def save_map_binary(map_data, output_file):
    trajectory_length = 91
    """Saves map data in a binary format readable by C"""
    with open(output_file, "wb") as f:
        # Count total entities
        print(len(map_data.get("objects", [])))
        print(len(map_data.get("roads", [])))
        num_objects = len(map_data.get("objects", []))
        num_roads = len(map_data.get("roads", []))
        # num_entities = num_objects + num_roads
        f.write(struct.pack("i", num_objects))
        f.write(struct.pack("i", num_roads))
        # f.write(struct.pack('i', num_entities))
        # Write objects
        for obj in map_data.get("objects", []):
            # Write base entity data
            obj_type = obj.get("type", 1)
            if obj_type == "vehicle":
                obj_type = 1
            elif obj_type == "pedestrian":
                obj_type = 2
            elif obj_type == "cyclist":
                obj_type = 3
            f.write(struct.pack("i", obj_type))  # type
            # f.write(struct.pack("i", obj.get("id", 0)))  # id
            f.write(struct.pack("i", trajectory_length))  # array_size
            # Write position arrays
            positions = obj.get("position", [])
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("x", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("y", 0.0))))
            for i in range(trajectory_length):
                pos = positions[i] if i < len(positions) else {"x": 0.0, "y": 0.0, "z": 0.0}
                f.write(struct.pack("f", float(pos.get("z", 0.0))))

            # Write velocity arrays
            velocities = obj.get("velocity", [])
            for arr, key in [(velocities, "x"), (velocities, "y"), (velocities, "z")]:
                for i in range(trajectory_length):
                    vel = arr[i] if i < len(arr) else {"x": 0.0, "y": 0.0, "z": 0.0}
                    f.write(struct.pack("f", float(vel.get(key, 0.0))))

            # Write heading and valid arrays
            headings = obj.get("heading", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}f",
                    *[float(headings[i]) if i < len(headings) else 0.0 for i in range(trajectory_length)],
                )
            )

            valids = obj.get("valid", [])
            f.write(
                struct.pack(
                    f"{trajectory_length}i",
                    *[int(valids[i]) if i < len(valids) else 0 for i in range(trajectory_length)],
                )
            )

            # Write scalar fields
            f.write(struct.pack("f", float(obj.get("width", 0.0))))
            f.write(struct.pack("f", float(obj.get("length", 0.0))))
            f.write(struct.pack("f", float(obj.get("height", 0.0))))
            goal_pos = obj.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", obj.get("mark_as_expert", 0)))

        # Write roads
        for idx, road in enumerate(map_data.get("roads", [])):
            geometry = road.get("geometry", [])
            road_type = road.get("map_element_id", 0)
            road_type_word = road.get("type", 0)
            if road_type_word == "lane":
                road_type = 2
            elif road_type_word == "road_edge":
                road_type = 15
            # breakpoint()
            if len(geometry) > 10 and road_type <= 16:
                geometry = simplify_polyline(geometry, 0.1)
            size = len(geometry)
            # breakpoint()
            if road_type >= 0 and road_type <= 3:
                road_type = 4
            elif road_type >= 5 and road_type <= 13:
                road_type = 5
            elif road_type >= 14 and road_type <= 16:
                road_type = 6
            elif road_type == 17:
                road_type = 7
            elif road_type == 18:
                road_type = 8
            elif road_type == 19:
                road_type = 9
            elif road_type == 20:
                road_type = 10
            # Write base entity data
            f.write(struct.pack("i", road_type))  # type
            # f.write(struct.pack("i", road.get("id", 0)))  # id
            f.write(struct.pack("i", size))  # array_size

            # Write position arrays
            for coord in ["x", "y", "z"]:
                for point in geometry:
                    f.write(struct.pack("f", float(point.get(coord, 0.0))))
            # Write scalar fields
            f.write(struct.pack("f", float(road.get("width", 0.0))))
            f.write(struct.pack("f", float(road.get("length", 0.0))))
            f.write(struct.pack("f", float(road.get("height", 0.0))))
            goal_pos = road.get("goalPosition", {"x": 0, "y": 0, "z": 0})  # Get goalPosition object with default
            f.write(struct.pack("f", float(goal_pos.get("x", 0.0))))  # Get x value
            f.write(struct.pack("f", float(goal_pos.get("y", 0.0))))  # Get y value
            f.write(struct.pack("f", float(goal_pos.get("z", 0.0))))  # Get z value
            f.write(struct.pack("i", road.get("mark_as_expert", 0)))


def load_map(map_name, binary_output=None):
    """Loads a JSON map and optionally saves it as binary"""
    with open(map_name, "r") as f:
        map_data = json.load(f)

    if binary_output:
        save_map_binary(map_data, binary_output)


def process_all_maps():
    """Process all maps and save them as binaries"""
    from pathlib import Path

    # Create the binaries directory if it doesn't exist
    binary_dir = Path("resources/drive/binaries")
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Path to the training data
    data_dir = Path("data/processed/training")

    # Get all JSON files in the training directory
    json_files = sorted(data_dir.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    # Process each JSON file
    for i, map_path in enumerate(json_files[:10000]):
        binary_file = f"map_{i:03d}.bin"  # Use zero-padded numbers for consistent sorting
        binary_path = binary_dir / binary_file

        print(f"Processing {map_path.name} -> {binary_file}")
        # try:
        load_map(str(map_path), str(binary_path))
        # except Exception as e:
        #     print(f"Error processing {map_path.name}: {e}")


def test_performance(timeout=10, atn_cache=1024, num_agents=1024):
    import time

    env = Drive(num_agents=num_agents)
    env.reset()
    tick = 0
    num_agents = 1024
    actions = np.stack(
        [np.random.randint(0, space.n + 1, (atn_cache, num_agents)) for space in env.single_action_space], axis=-1
    )

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {num_agents * tick / (time.time() - start)}")
    env.close()


if __name__ == "__main__":
    # test_performance()
    process_all_maps()
