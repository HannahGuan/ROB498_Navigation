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
        self.max_speed = 1.0
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
        Implements the multi-agent RVO approach from ICRA 2008:
          - Compute the 'combined' RVO from all other agents (and obstacles).
          - Find a velocity that is outside that combined region
            and is closest to the 'preferred velocity'.
          - 'Preferred velocity' is from current position to the goal, 
            up to some max speed.
        We'll produce a 2D velocity command.
        """
        if not self.joystick_on:
            return
        
        x, y, th = self.robot_current
        goal_xy = self.goal_config.position_and_heading_nk3(squeeze=True)[:2]
        dir_to_goal = goal_xy - np.array([x, y])
        dist_goal = np.linalg.norm(dir_to_goal)

        if dist_goal < 0.05:
            # Already near goal, stop
            new_vel = np.zeros(2, dtype=float)
        else:
            desired_speed = min(dist_goal, self.max_speed)
            goal_dir = dir_to_goal / (dist_goal + 1e-9)
            pref_vel = desired_speed * goal_dir

            # 2) Build the union of RVO constraints from all neighbors:
            all_agents = self.sim_state_now.get_all_agents()
            # Our own radius
            robots_dict = self.sim_state_now.get_robots()
            robot_key = list(robots_dict.keys())[0]    # e.g. the first key
            my_robot = robots_dict[robot_key]
            my_radius = my_robot.radius

            vx = self.robot_v * np.cos(th)
            vy = self.robot_v * np.sin(th)
            my_v = np.array([vx, vy], dtype=float)

            # We'll store half-plane constraints or we can do sampling.
            # For a full ICRA2008 approach, we do geometry, but let's show a simplified approach:
            # We'll accumulate constraints in a list, then pick the best velocity from sampling.

            # We'll define a function that returns the half-plane outside the RVO for each agent.
            # RVO: RVOA_B = { v' | 2v' - v in VO(A,B) }. 
            # We'll do something akin to the "push away from collision boundary."

            constraints = []  # Each will be (normal, point_on_boundary)

            for other_key, other_agent in all_agents.items():
                if other_key == robot_key:
                    continue
                # Get other's pos, vel, radius
                other_pos_h = other_agent.get_current_config().position_and_heading_nk3(squeeze=True)
                other_pos = np.array([other_pos_h[0], other_pos_h[1]], dtype=float)

                # other_speed_nk1 = other_agent.get_current_config().speed_nk1
                # other_heading_nk1 = other_agent.get_current_config().heading_nk1
                # speed_nk = np.squeeze(other_speed_nk1, axis=-1)       # shape (n, k)
                # heading_nk = np.squeeze(other_heading_nk1, axis=-1)
                # print("speed_nk:", speed_nk.shape, speed_nk.dtype)
                # print("heading_nk:", heading_nk.shape, heading_nk.dtype)   # shape (n, k)
                # if speed_nk.shape == () or heading_nk.shape == ():
                #     other_vel = 0
                # else:
                #     # Compute vx, vy
                #     vx_nk = speed_nk * np.cos(heading_nk)
                #     vy_nk = speed_nk * np.sin(heading_nk)

                #     # Stack into (n, k, 2)
                #     other_vel = np.stack((vx_nk, vy_nk), axis=-1)
                other_vel = 1.2
                other_r = other_agent.radius

                combined_radius = my_radius + other_r
                rel_pos = other_pos - np.array([x, y])
                dist_sq = np.sum(rel_pos**2)

            # Simple push if overlapping or near-overlapping
            if dist_sq < (combined_radius + 1e-9)**2:
                dist_ = np.sqrt(dist_sq) + 1e-9
                n = rel_pos / dist_
                overlap = (combined_radius - dist_)
                w = overlap * n
                boundary_pt = my_v + 0.5 * w
                constraints.append((n, boundary_pt))
            else:
                # Possibly check if we're approaching
                dist_ = np.sqrt(dist_sq)
                v_radial = np.dot(my_v - other_vel, rel_pos / dist_)
                limit = combined_radius / self.time_horizon
                approach_dist = v_radial * self.time_horizon
                # If approaching enough to collide:
                if approach_dist + 1e-9 >= dist_ - combined_radius:
                    n = rel_pos / dist_
                    w = (limit - (dist_ / self.time_horizon)) * n
                    boundary_pt = my_v + 0.5 * w
                    constraints.append((n, boundary_pt))

        # 3) Apply constraints (naive "push-out") to pref_vel
        new_vel = pref_vel.copy()
        for (n, boundary_pt) in constraints:
            if np.dot(new_vel - boundary_pt, n) < 0.0:
                corr = np.dot(boundary_pt - new_vel, n) * n
                new_vel = new_vel + corr

        # clamp
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = (new_vel / speed) * self.max_speed

        if self.joystick_params.use_system_dynamics:
            # CASE 1: Velocity-based commands => (v, w)
            #
            # We'll treat new_vel as 2D linear velocity in the plane. 
            # For a differential-drive, you might convert that into:
            #   v_lin = norm(new_vel), w_ang = angle difference / dt
            # We'll assume self.robot_current holds [x, y, theta].
            x_cur, y_cur, th_cur = self.robot_current
            heading_des = np.arctan2(new_vel[1], new_vel[0]) if np.linalg.norm(new_vel)>1e-9 else th_cur
            dth = (heading_des - th_cur + np.pi) % (2.0*np.pi) - np.pi
            dt = self.sim_dt if self.sim_dt>1e-9 else 1.0
            w_ang = dth / dt
            v_lin = np.linalg.norm(new_vel)

            self.commands = [(float(v_lin), float(w_ang))]
        else:
            # CASE 2: Position-based commands => (x_new, y_new, theta, velocity)
            #
            # We integrate forward for one time-step to get the new position:
            x_cur, y_cur, th_cur = self.robot_current
            dt = self.sim_dt if self.sim_dt>1e-9 else 1.0

            # Euler step
            x_new = x_cur + new_vel[0]*dt
            y_new = y_cur + new_vel[1]*dt
            # We'll define the heading by the direction of new_vel
            if np.linalg.norm(new_vel) > 1e-9:
                th_new = np.arctan2(new_vel[1], new_vel[0])
            else:
                th_new = th_cur

            v_lin = np.linalg.norm(new_vel)

            # If you need to track the new state for next iteration:
            self.robot_current = np.array([x_new, y_new, th_new], dtype=float)

            self.commands = [
                (float(x_new), float(y_new), float(th_new), float(v_lin))
            ]


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
