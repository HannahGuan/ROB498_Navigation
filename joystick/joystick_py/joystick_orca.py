from random import randint
from typing import List, Tuple, Optional
from dotmap import DotMap
from params.central_params import (
    create_agent_params,
    create_joystick_params,
    create_system_dynamics_params,
)
from agents.agent import Agent
from obstacles.sbpd_map import SBPDMap
from socnav.socnav_renderer import SocNavRenderer  # if you want to visualize
from trajectory.trajectory import SystemConfig
from utils.utils import euclidean_dist2
from objectives.objective_function import ObjectiveFunction
from objectives.personal_space_cost import PersonalSpaceCost

import numpy as np

from joystick_py.joystick_base import JoystickBase

class JoystickORCA(JoystickBase):
    def __init__(self):
        self.robot_current: np.ndarray = None  # current position of the robot
        self.robot_v = 0.0           # Current linear speed
        self.robot_w = 0.0           # Current angular speed
        super().__init__("ORCAPlanner")
        # Parameters
        self.max_speed = 1.2
        self.agent_hist = {}
        self.time_horizon = 2.0  # e.g., how far ahead we plan to avoid collisions
        self.commands = []

    def init_obstacle_map(self) -> SBPDMap:
        """
        Similar to joystick_planner.py, you can create an SBPDMap from your environment.
        This then can be used to measure distances to obstacles for repulsive forces.
        """
        # We rely on self.current_ep being set up by the base class after we get episode data.
        # environment is basically a dictionary with "map_traversible", "map_scale", etc.
        env = self.current_ep.get_environment()
        p = self.agent_params.obstacle_map_params
        # We can pass a renderer if we want visual debug, or just None/0:
        renderer = SocNavRenderer() if p.render_2D else 0

        sbpd_map = p.obstacle_map(
            p,
            renderer,
            res=float(env["map_scale"]) * 100.0,
            map_trav=np.array(env["map_traversible"]),
        )
        return sbpd_map
    

    def init_control_pipeline(self) -> None:
        """
        Called once the base class has the episode info. We'll do what joystick_planner does:
          - set up agent_params
          - build the SBPDMap
        """
        self.start_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_start())
        self.goal_config: SystemConfig = SystemConfig.from_pos3(self.get_robot_goal())
        # rest of the 'Agent' params used for the joystick planner
        self.agent_params: DotMap = create_agent_params(
            with_planner=True, with_obstacle_map=True
        )
        # update generic 'Agent params' with joystick-specific params
        self.agent_params.episode_horizon_s = self.joystick_params.episode_horizon_s
        self.agent_params.control_horizon_s = self.joystick_params.control_horizon_s
        # init obstacle map
        self.obstacle_map: SBPDMap = self.init_obstacle_map()
        self.obj_fn: ObjectiveFunction = Agent._init_obj_fn(
            self, params=self.agent_params
        )
        psc_obj = PersonalSpaceCost(params=self.agent_params.personal_space_objective)
        self.obj_fn.add_objective(psc_obj)

        # Initialize Fast-Marching-Method map for agent's pathfinding
        Agent._init_fmm_map(self, params=self.agent_params)

        # Initialize system dynamics and planner fields
        self.planner = Agent._init_planner(self, params=self.agent_params)
        self.vehicle_data = self.planner.empty_data_dict()
        self.system_dynamics = Agent._init_system_dynamics(
            self, params=self.agent_params
        )
        # init robot current config from the starting position
        self.robot_current = self.current_ep.get_robot_start().copy()
        # init a list of commands that will be sent to the robot
        self.commands = None

    def joystick_sense(self) -> None:
        # ping the robot
        self.send_to_robot("sense")

        # optionally store the old position
        robot_prev = None
        if self.robot_current is not None:
            robot_prev = self.robot_current.copy()

        # get updated sim_state
        self.joystick_on = self.listen_once()
        if not self.joystick_on:
            return

        # now update robot_current properly
        robot = list(self.sim_state_now.get_robots().values())[0]
        self.robot_current = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )
        # Updating robot speeds (linear and angular) based off simulator data
        if robot_prev is not None:
            self.robot_v = euclidean_dist2(self.robot_current, robot_prev) / self.sim_dt
            self.robot_w = (self.robot_current[2] - robot_prev[2]) / self.sim_dt
        else:
            self.robot_v = 0
            self.robot_w = 0

    def joystick_plan(self) -> None:
        """
        Implements a more "turn-friendly" ORCA approach:
        1) Compute the preferred 2D velocity toward the goal.
        2) Build ORCA constraints from neighbors/obstacles; adjust velocity.
        3) Convert the resulting 2D velocity to differential-drive (v, w),
            allowing enough angular speed to turn promptly.
        """
        if not self.joystick_on:
            return

        # -- Retrieve current robot pose --
        x, y, th = self.robot_current  # [x, y, theta]

        # -- Compute direction to goal --
        goal_xy = self.goal_config.position_and_heading_nk3(squeeze=True)[:2]
        dir_to_goal = goal_xy - np.array([x, y])
        dist_goal = np.linalg.norm(dir_to_goal)

        # Small threshold to decide if we're "close enough" to goal
        if dist_goal < 0.05:
            # Stop if near the goal
            new_vel = np.zeros(2, dtype=float)
        else:
            # 1) Preferred velocity = direction to goal, up to self.max_speed
            desired_speed = min(dist_goal, self.max_speed)
            goal_dir = dir_to_goal / (dist_goal + 1e-9)
            pref_vel = desired_speed * goal_dir

            # 2) Build ORCA constraints by checking neighbors
            all_agents = self.sim_state_now.get_all_agents()
            my_robot = list(self.sim_state_now.get_robots().values())[0]
            my_radius = my_robot.radius

            # Current velocity in 2D (based on v, w)
            vx = self.robot_v * np.cos(th)
            vy = self.robot_v * np.sin(th)
            my_v = np.array([vx, vy], dtype=float)

            constraints = []  # Will store (normal, boundary_point) half-plane constraints

            
            for other_key, other_agent in all_agents.items():
                if other_agent is my_robot:
                    continue
                other_pos_h = other_agent.get_current_config().position_and_heading_nk3(
                    squeeze=True
                )
                other_pos = np.array([other_pos_h[0], other_pos_h[1]], dtype=float)
                
                # Current config for this agent at this timestep
                current_config = other_agent.get_current_config()

                # Build a key to store its previous config in a dictionary
                agent_prev_str = other_key + "_prev"
                # If we have no previous config, initialize it
                if agent_prev_str not in self.agent_hist:
                    self.agent_hist[agent_prev_str] = current_config

                # Extract [x, y, theta] from the current config (shape: (3,))
                pos_and_heading_now = current_config.position_and_heading_nk3(squeeze=True)
                pos_now = pos_and_heading_now[:2]  # (x, y)

                # Extract [x, y, theta] from the previous config
                prev_config = self.agent_hist[agent_prev_str]
                pos_and_heading_prev = prev_config.position_and_heading_nk3(squeeze=True)
                pos_prev = pos_and_heading_prev[:2]  # (x, y)

                # Compute delta time for this step
                dt = self.sim_dt

                # Compute velocities in x, y
                vx = (pos_now[0] - pos_prev[0]) / dt
                vy = (pos_now[1] - pos_prev[1]) / dt

                other_vel = np.array([vx, vy], dtype=float)
                # print(f"Agent {other_key} velocity = ({vx:.3f}, {vy:.3f})")
                # print(f"Agent {other_key} position = ({pos_now[0]:.3f}, {pos_prev[0]:.3f})")

                # Update the stored "previous config" to the current config
                self.agent_hist[agent_prev_str] = current_config

                other_r = other_agent.radius
                combined_radius = my_radius + other_r

                rel_pos = other_pos - np.array([x, y])
                dist_sq = np.sum(rel_pos**2)

                if dist_sq < (combined_radius + 1e-9) ** 2:
                    # Overlap or near-overlap => push away
                    dist_ = np.sqrt(dist_sq) + 1e-9
                    n = rel_pos / dist_
                    overlap = combined_radius - dist_
                    w = overlap * n
                    boundary_pt = my_v + 0.5 * w
                    constraints.append((n, boundary_pt))
                else:
                    # Check if approaching collision in the time horizon
                    dist_ = np.sqrt(dist_sq)
                    n = rel_pos / dist_
                    v_radial = np.dot(my_v - other_vel, n)
                    limit = combined_radius / self.time_horizon
                    approach_dist = v_radial * self.time_horizon
                    # If we will collide within time_horizon, add constraint
                    if approach_dist + 1e-9 >= dist_ - combined_radius:
                        w = (limit - (dist_ / self.time_horizon)) * n
                        boundary_pt = my_v + 0.5 * w
                        constraints.append((n, boundary_pt))

            # 3) Apply constraints ("push-out" approach) to the preferred velocity
            new_vel = pref_vel.copy()
            for (n, boundary_pt) in constraints:
                if np.dot(new_vel - boundary_pt, n) < 0.0:
                    # Project new_vel onto boundary
                    corr = np.dot(boundary_pt - new_vel, n) * n
                    new_vel = new_vel + corr

            # 4) Clamp to max_speed
            speed = np.linalg.norm(new_vel)
            if speed > self.max_speed:
                new_vel = (new_vel / (speed + 1e-9)) * self.max_speed

        # -- Convert the resulting 2D velocity into (v, w) or (x,y,theta,velocity) --

        if self.joystick_params.use_system_dynamics:
            #
            # CASE 1: (v, w) for a differential-drive robot
            #
            # Let new_vel = (vx, vy) in the world frame
            dt = self.sim_dt if self.sim_dt > 1e-9 else 0.1
            vx_des, vy_des = new_vel
            desired_heading = np.arctan2(vy_des, vx_des) if np.linalg.norm(new_vel) > 1e-9 else th

            # Angular difference
            dtheta = desired_heading - th
            # Wrap to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

            # Example maximum turn rate (could be a parameter)
            max_angular_speed = 1.5  # rad/s
            w_cmd = np.clip(dtheta / dt, -max_angular_speed, max_angular_speed)

            # Forward speed is the norm of new_vel, but we can reduce if turning
            # For instance, if you want to slow down when angle is large:
            #   forward_scale = max(0.0, np.cos(dtheta))
            #   v_cmd = speed * forward_scale
            # Or you can just pass speed as is:
            v_cmd = np.linalg.norm(new_vel)

            self.commands = [(float(v_cmd), float(w_cmd))]

        else:
            #
            # CASE 2: Position-based command => (x_new, y_new, theta_new, speed)
            #
            dt = self.sim_dt if self.sim_dt > 1e-9 else 0.1
            vx_des, vy_des = new_vel
            x_new = x + vx_des * dt
            y_new = y + vy_des * dt

            if np.linalg.norm(new_vel) > 1e-9:
                th_new = np.arctan2(vy_des, vx_des)
            else:
                th_new = th

            v_cmd = np.linalg.norm(new_vel)
            self.robot_current = np.array([x_new, y_new, th_new], dtype=float)

            self.commands = [(float(x_new), float(y_new), float(th_new), float(v_cmd))]


    def joystick_act(self) -> None:
        """
        Send out the velocity commands as (v, w), similar to joystick_random or planner code.
        """
        if not self.joystick_on or not self.commands:
            return

        self.send_cmds(
            self.commands,
            send_vel_cmds=self.joystick_params.use_system_dynamics
        )
        self.commands = []

    def update_loop(self) -> None:
        """
        This is basically the same pattern used by joystick_random and joystick_planner:
          - pre_update() sets up the socket listening, etc.
          - while loop does sense -> plan -> act
          - finish_episode() at the end
        """
        super().pre_update()
        self.simulator_joystick_update_ratio = int(
            np.floor(self.sim_dt / self.agent_params.dt)
        )
        while self.joystick_on:
            self.joystick_sense()
            self.joystick_plan()
            self.joystick_act()
        self.finish_episode()
