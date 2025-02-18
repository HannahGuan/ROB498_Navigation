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


class JoystickSocialForce(JoystickBase):
    def __init__(self):
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current: np.ndarray = None  # current position of the robot
        self.robot_v = 0.0           # Current linear speed
        self.robot_w = 0.0           # Current angular speed
        super().__init__("SocialForcePlanner")  # parent class needs to know the algorithm

        self.agent_params = None
        self.obstacle_map = None

        self.relaxation_time = 0.5    # tau
        self.desired_speed = 1.3     # typical pedestrian speed (m/s)
        self.V0 = 2.1                # repulsive strength
        self.sigma = 0.3             # range parameter for exponential
        self.commands = []
        self.simulator_joystick_update_ratio: int = 1

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

        # Create the agent_params with obstacle map turned on, if you like:
        self.agent_params = create_agent_params(
            with_planner=False,      # we won't do the sampling planner
            with_obstacle_map=True,  # we do want map info
        )

        # Additional param tweaks:
        self.agent_params.control_horizon_s = self.joystick_params.control_horizon_s
        self.agent_params.episode_horizon_s = self.joystick_params.episode_horizon_s

        # Build the SBPDMap to measure obstacle distances for repulsive forces:
        self.obstacle_map = self.init_obstacle_map()
        self.obj_fn: ObjectiveFunction = Agent._init_obj_fn(
            self, params=self.agent_params
        )
        psc_obj = PersonalSpaceCost(params=self.agent_params.personal_space_objective)
        self.obj_fn.add_objective(psc_obj)

        # Initialize Fast-Marching-Method map for agent's pathfinding
        Agent._init_fmm_map(self, params=self.agent_params)

        # Initialize system dynamics and planner fields
        # self.planner = Agent._init_planner(self, params=self.agent_params)
        # self.vehicle_data = self.planner.empty_data_dict()
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
        Core Social Force computation:
          F_total = F_desired + sum(F_repulsion from other agents) + sum(F_repulsion from obstacles)
          -> integrate to get new velocity
        Then we store the results in self.commands for joystick_act() to send.
        """
        if not self.joystick_on:
            return

        # 1) Desired velocity ~ heading to goal:
        robot_pos = self.robot_current
        robot_vel = np.array([self.robot_v, self.robot_w])  # or store as full vx, vy

        # Get the final goal from the Episode:
        goal_xyz = self.current_ep.get_robot_goal()  # [gx, gy, gtheta]
        # compute direction in 2D
        goal_dir = goal_xyz[:2] - robot_pos[:2]
        dist_to_goal = np.linalg.norm(goal_dir)
        if dist_to_goal > 1e-5:
            goal_dir /= dist_to_goal
        else:
            goal_dir = np.array([0., 0.])

        # The "desired" speed could be self.desired_speed or vary with dist_to_goal
        v_des_2d = goal_dir * self.desired_speed

        # (a) F_desired = (v_des - v_actual) / tau
        # If your actual velocity is purely linear in the heading direction, you can do:
        #   v_actual_2d = [v*cos(th), v*sin(th)]
        th = robot_pos[2]
        v_actual_2d = np.array([self.robot_v * np.cos(th),
                                self.robot_v * np.sin(th)])
        F_desired = (v_des_2d - v_actual_2d) / self.relaxation_time

        # 2) Repulsion from other agents:
        F_agents = np.zeros(2)
        all_peds = self.sim_state_now.get_all_agents()
        # This is your ID in the dictionary:
        robot_key = list(self.sim_state_now.get_robots().keys())[0]
        for key, agent in all_peds.items():
            if key == robot_key:
                continue
            pos_other = agent.get_current_config().position_and_heading_nk3(squeeze=True)
            diff = robot_pos[:2] - pos_other[:2]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue
            # Helbing’s exponential:
            direction = diff / dist
            rep_val = self.V0 * np.exp(-dist / self.sigma)
            F_agents += rep_val * direction

        # 3) Repulsion from obstacles via self.obstacle_map
        #    For each step, find distance to nearest obstacle/wall, do an exponential push
        #    The SBPDMap has methods like get_signed_distance(). You’d integrate that:
        # 3) Repulsion from obstacles using SBPDMap.dist_to_nearest_obs(...) 
        #    We'll approximate the distance gradient via small finite differences around (x, y).
        eps = 0.05
        x, y = robot_pos[0], robot_pos[1]

        # Distance at the robot's current position:
        pos_center = np.array([[[x, y]]], dtype=float)  # shape (1,1,2)
        dist_val = self.obstacle_map.dist_to_nearest_obs(pos_center)[0, 0]

        # Distance at x+eps, x-eps
        pos_plus_x = np.array([[[x + eps, y]]], dtype=float)
        pos_minus_x = np.array([[[x - eps, y]]], dtype=float)
        dist_plus_x = self.obstacle_map.dist_to_nearest_obs(pos_plus_x)[0, 0]
        dist_minus_x = self.obstacle_map.dist_to_nearest_obs(pos_minus_x)[0, 0]
        df_dx = (dist_plus_x - dist_minus_x) / (2.0 * eps)

        # Distance at y+eps, y-eps
        pos_plus_y = np.array([[[x, y + eps]]], dtype=float)
        pos_minus_y = np.array([[[x, y - eps]]], dtype=float)
        dist_plus_y = self.obstacle_map.dist_to_nearest_obs(pos_plus_y)[0, 0]
        dist_minus_y = self.obstacle_map.dist_to_nearest_obs(pos_minus_y)[0, 0]
        df_dy = (dist_plus_y - dist_minus_y) / (2.0 * eps)

        # Normalize the gradient, if non-zero:
        grad_d = np.array([df_dx, df_dy], dtype=float)
        grad_norm = np.linalg.norm(grad_d)
        if grad_norm > 1e-9:
            grad_d /= grad_norm

        # Helbing-style exponential repulsion using nonnegative distance:
        rep_obs = self.V0 * np.exp(-dist_val / self.sigma)
        F_walls = rep_obs * grad_d


        # 4) Sum the forces:
        F_total_2d = F_desired + F_agents + F_walls

        # Integrate for dt => new velocity
        dt = self.sim_dt
        v_new_2d = v_actual_2d + F_total_2d * dt
        # clamp speed if needed:
        max_speed = self.system_dynamics_params.v_bounds[1]
        speed_new = np.linalg.norm(v_new_2d)
        if speed_new > max_speed:
            v_new_2d *= (max_speed / speed_new)

        # Convert (vx, vy) -> (v_lin, w_ang) for velocity-based or -> (x_new, y_new, th_new, speed_new) for position-based
        new_heading = np.arctan2(v_new_2d[1], v_new_2d[0])
        dtheta = (new_heading - th + np.pi) % (2.0 * np.pi) - np.pi
        v_lin = np.linalg.norm(v_new_2d)
        w_ang = dtheta / dt

        # If position-based, we should compute the next (x, y, theta, speed)
        # Basic Euler step: x_new = x + vx*dt, y_new = y + vy*dt
        #   (where vx, vy = v_new_2d)
        x_new = x + v_new_2d[0] * dt
        y_new = y + v_new_2d[1] * dt
        th_new = new_heading
        # velocity param is v_lin if your code wants that

        # Suppose self.goal_config is a SystemConfig.
        # We'll extract goal [gx, gy, gtheta], or just [gx, gy].
        goal_posn_heading = self.goal_config.position_and_heading_nk3(squeeze=True)
        # This should return a numpy array of shape (3,). Then we can do:
        goal_xy = goal_posn_heading[:2]
        dist = np.linalg.norm(np.array([x_new, y_new]) - goal_xy)
        print("GOAL DIST: " + str(dist))
        if dist < 0.1:
            if self.joystick_params.use_system_dynamics:
                # velocity-based: set (v, w) = (0, 0)
                self.commands = [(0.0, 0.0)]
            else:
                # position-based: send the same position
                x, y, th = robot_pos
                self.commands = [(x, y, th, 0.0)]
            return

        if self.joystick_params.use_system_dynamics:
            # (v, w)
            self.commands = [(float(v_lin), float(w_ang))]
        else:
            # (x, y, theta, velocity)
            self.commands = [(float(x_new), float(y_new), float(th_new), float(v_lin))]

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
        # self.simulator_joystick_update_ratio = int(
        #     np.floor(self.sim_dt / self.agent_params.joystick_params.dt)
        # )
        while self.joystick_on:
            self.joystick_sense()
            self.joystick_plan()
            self.joystick_act()
        self.finish_episode()


    
