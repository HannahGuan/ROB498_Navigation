from typing import List, Optional

import numpy as np
from agents.agent import Agent
from dotmap import DotMap
from objectives.objective_function import ObjectiveFunction
from objectives.personal_space_cost import PersonalSpaceCost
from obstacles.sbpd_map import SBPDMap
from params.central_params import create_agent_params
from socnav.socnav_renderer import SocNavRenderer
from trajectory.trajectory import SystemConfig, Trajectory
from utils.utils import euclidean_dist2

from joystick_py.joystick_base import JoystickBase

from . import brne


class JoystickBRNE(JoystickBase):
    def __init__(self):
        # planner variables
        # the list of commands sent to the robot to execute
        self.commands: List[str] = []
        self.simulator_joystick_update_ratio: int = 1
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current: np.ndarray = None  # current position of the robot
        self.robot_v: float = 0  # not tracked in the base simulator
        self.robot_w: float = 0  # not tracked in the base simulator
        super().__init__("BRNE")  # parent class needs to know the algorithm

        print('use system dynamics: ', self.joystick_params.use_system_dynamics)
        assert not self.joystick_params.use_system_dynamics

        self.x_list: np.ndarray = None
        self.y_list: np.ndarray = None
        self.th_list: np.ndarray = None
        self.v_list: np.ndarray = None
    
    def from_conf(self, configs, idx):
        x = float(configs._position_nk2[0][idx][0])
        y = float(configs._position_nk2[0][idx][1])
        th = float(configs._heading_nk1[0][idx][0])
        v = float(configs._speed_nk1[0][idx][0])
        return (x, y, th, v)

    def init_obstacle_map(self, renderer: Optional[SocNavRenderer] = 0) -> SBPDMap:
        """ Initializes the sbpd map."""
        p: DotMap = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(
            p,
            renderer,
            res=float(env["map_scale"]) * 100.0,
            map_trav=np.array(env["map_traversible"]),
        )

    def init_control_pipeline(self) -> None:
        # NOTE: this is like an init() run *after* obtaining episode metadata
        # robot start and goal to satisfy the old Agent.planner
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

        #################################################
        self.tsteps = 10
        self.num_peds = 8
        self.num_pts = 300
        self.num_steps = 10

        self.robot = self.get_robot_start()
        self.agents = {}
        agents_info = self.current_ep.get_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
        
        # sim_tlist = np.arange(self.tsteps) * self.sim_dt
        

    def joystick_sense(self):
        # ping's the robot to request a sim state
        self.send_to_robot("sense")

        # store previous pos3 of the robot (x, y, theta)
        robot_prev = self.robot_current.copy()  # copy since its just a list
        # listen to the robot's reply
        self.joystick_on = self.listen_once()

        # NOTE: at this point, self.sim_state_now is updated with the
        # most up-to-date simulation information

        # Update robot current position
        robot = list(self.sim_state_now.get_robots().values())[0]
        self.robot_current = robot.get_current_config().position_and_heading_nk3(
            squeeze=True
        )

        # Updating robot speeds (linear and angular) based off simulator data
        self.robot_v = euclidean_dist2(self.robot_current, robot_prev) / self.sim_dt
        self.robot_w = (self.robot_current[2] - robot_prev[2]) / self.sim_dt

        #################################
        robot_prev = self.robot.copy()
        agents_prev = {}
        for key in list(self.agents.keys()):
            agent = self.agents[key]
            agents_prev[key] = agent.copy()

        self.agents = {}
        self.agents_radius = {}
        agents_info = self.sim_state_now.get_all_agents()
        for key in list(agents_info.keys()):
            agent = agents_info[key]
            self.agents[key] = np.squeeze(
                agent.get_current_config().position_and_heading_nk3()
            )
            self.agents_radius[key] = agent.get_radius()
        robot_tmp = list(self.sim_state_now.get_robots().values())[0]
        self.robot = np.squeeze(
            robot_tmp.get_current_config().position_and_heading_nk3()
        )
        self.robot_radius = robot_tmp.get_radius()

        # self.robot_v = (self.robot - robot_prev) / self.sim_dt
        self.agents_v = {}
        for key in list(self.agents.keys()):
            if key in agents_prev:
                v = (self.agents[key] - agents_prev[key]) / self.sim_dt / 10
            else:
                v = np.array([0, 0, 0], dtype=np.float32)
            self.agents_v[key] = v

    def joystick_plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        - Access to sim_states from the self.current_world
        """
        # get information about robot by its "current position" which was updated in sense()
        [x, y, th] = self.robot_current
        v = self.robot_v
        # can also try:
        #     # assumes the robot has executed all the previous commands in self.commands
        #     (x, y, th, v) = self.from_conf(self.commands, -1)
        robot_config = SystemConfig.from_pos3(pos3=(x, y, th), v=0.3)
        self.planner_data = self.planner.optimize(
            robot_config, self.goal_config, sim_state_hist=self.sim_states
        )

        agents_info = self.get_proper_pedestrain_data(self.agents)
        self.history_traj_library(self.agents)
        self.llm_tool.set_history_library(self.history)

        goal = self.get_robot_goal()
        dist2goal = np.sqrt((x-goal[0])**2 + (y-goal[1])**2)

        # TODO: make sure the planning control horizon is greater than the
        # simulator_joystick_update_ratio else it will not plan far enough

        tsteps = self.tsteps
        tlist = np.arange(tsteps) * self.sim_dt
        v_nominal = np.array([goal[0]-x, goal[1]-y])
        v_nominal = v_nominal / np.sqrt(v_nominal[0]**2+v_nominal[1]**2)
        v_nominal *= 1.0
        inc_list = tlist * v_nominal[:,np.newaxis]
        x_list = x + inc_list[0]
        y_list = y + inc_list[1]

        dist2obst = []
        for xt, yt in zip(x_list[:self.num_steps], y_list[:self.num_steps]):
            dist2obst.append(self.obstacle_map.dist_to_nearest_obs(np.array([[[xt,yt]]])))
        # print('dist to obst: ', np.min(dist2obst))
        min_dist2obst = np.min(dist2obst)
        
        tsteps = self.tsteps
        tlist = np.arange(tsteps) * self.sim_dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-02])
        test_ts = tlist
        self.cov_Lmat, cov_mat = brne.get_Lmat_nb(train_ts, test_ts, train_noise)
        # print('cov diag: ', np.diagonal(cov_mat)[:10], end='  ')

        agent_dist_list = np.zeros(len(self.agents))
        for i, key in enumerate(list(self.agents.keys())):
            agent_dist_list[i] = np.sqrt((x-self.agents[key][0])**2 + (y-self.agents[key][1])**2)
        
        ped_keys = []
        for _i in range(len(agent_dist_list)):
            if agent_dist_list[_i] < 3.5:
                ped_keys.append(list(self.agents.keys())[_i])
        # ped_keys_coarse = [list(self.agents.keys())[_i] for _i in np.argsort(agent_dist_list)[:self.num_peds]]
        num_brne_agents = len(ped_keys) + 1


        meta_flag = False
        if dist2goal > 1.0 and min_dist2obst < 0.3:
        # if True:
            meta_flag = True
            self.commands = Trajectory.new_traj_clip_along_time_axis(
                self.planner_data["trajectory"],
                # self.agent_params.control_horizon,
                10,
                repeat_second_to_last_speed=True,
            )
            x_list = np.array(self.commands._position_nk2[0][:,0])
            y_list = np.array(self.commands._position_nk2[0][:,1])
            # print('verify x_list: ', len(x_list), end='  ')
            x_opt_trajs = x_list.copy()
            y_opt_trajs = y_list.copy()
            x_list = x_opt_trajs.copy()
            y_list = y_opt_trajs.copy()
 
        elif num_brne_agents ==1:
            meta_flag = True
            self.commands = Trajectory.new_traj_clip_along_time_axis(
                self.planner_data["trajectory"],
                # self.agent_params.control_horizon,
                10,
                repeat_second_to_last_speed=True,
            )
            x_list = np.array(self.commands._position_nk2[0][:,0])
            y_list = np.array(self.commands._position_nk2[0][:,1])
            x_opt_trajs = x_list.copy()
            y_opt_trajs = y_list.copy()
            x_list = x_opt_trajs.copy()
            y_list = y_opt_trajs.copy()
        else:
            x_pts = brne.mvn_sample_normal(num_brne_agents * self.num_pts, tsteps, self.cov_Lmat)
            y_pts = brne.mvn_sample_normal(num_brne_agents * self.num_pts, tsteps, self.cov_Lmat)

            xmean_list = np.zeros((num_brne_agents, tsteps))
            ymean_list = np.zeros((num_brne_agents, tsteps))
            xmean_list[0] = x_list.copy()
            ymean_list[0] = y_list.copy()

            all_traj_pts_x = np.zeros((num_brne_agents * self.num_pts, tsteps))
            all_traj_pts_y = np.zeros((num_brne_agents * self.num_pts, tsteps))
            
            all_traj_pts_x[:self.num_pts ] = xmean_list[0] + x_pts[:self.num_pts ]
            all_traj_pts_y[:self.num_pts ] = ymean_list[0] + y_pts[:self.num_pts ]
            for i, key in enumerate(ped_keys):
                ped_v = np.array(self.agents_v[key][:2])
                ped_pos = self.agents[key]
                # print('ped_v: ', ped_v, end='  ')
                # ped_v /= np.sqrt(ped_v[0]**2 + ped_v[1]**2)
                # ped_v *= 0.1

                intention, result = self.llm_tool.get_one_prediction(key, agents_info, [x,y])
                try: 
                    left_ratio = round(result['left'],2)
                    right_ratio = round(result['right'],2)
                except Exception:
                    right_ratio = (1-0.7)/2
                    left_ratio = (1-0.7)/2
                # xmean_list[i+1] = self.agents[key][0] + (tlist) * ped_v[0]
                # ymean_list[i+1] = self.agents[key][1] + (tlist) * ped_v[1]
                d_x = x_pts[(i+1) * self.num_pts: (i+2) * self.num_pts + self.num_pts]
                d_y = y_pts[(i+1) * self.num_pts: (i+2) * self.num_pts + self.num_pts]
                x_sampled, y_sampled = self.generate_biased_trajectories(d_x,d_y, ped_pos, ped_v, self.num_pts, left_ratio, right_ratio)
                xmean_list[i+1] = np.mean(x_sampled, axis=0) 
                ymean_list[i+1] = np.mean(y_sampled, axis=0)
                all_traj_pts_x[(i+1) * self.num_pts: (i+2) * self.num_pts] = x_sampled
                all_traj_pts_y[(i+1) * self.num_pts: (i+2) * self.num_pts] = y_sampled
            
            x_opt_trajs = xmean_list.copy()
            y_opt_trajs = ymean_list.copy()

        if meta_flag == False:
        # if True:
            # if np.min(agent_dist_list) < 1.0:
            
            x_opt_trajs, y_opt_trajs, weights = brne.brne_nav(
                xmean_list, ymean_list, x_pts, y_pts,
                num_brne_agents, tsteps, self.num_pts,all_traj_pts_x, all_traj_pts_y
            )

            x_list = x_opt_trajs[0].copy()
            y_list = y_opt_trajs[0].copy()
            # print('weights: ', weights[0][::10], end='  ')


        v_list = np.sqrt((y_list[1:]-y_list[:-1])**2 + (x_list[1:]-x_list[:-1])**2)/0.1
        v_list = np.array([v, *v_list])
        
        th_list = np.arctan2(y_list[1:]-y_list[:-1], x_list[1:]-x_list[:-1])
        th_list = np.array([th, *th_list])

        for i in range(1, len(x_list)):
            delta_x = x_list[i] - x_list[i - 1]
            delta_y = y_list[i] - y_list[i - 1]
            distance = np.sqrt(delta_x**2 + delta_y**2)

            # 计算速度
            speed = distance / 0.1

            # 如果速度超过限制，调整当前点的位置
            if speed > 1.2:
                # 计算缩放因子
                scaling_factor = (1.2 * 0.1) / distance
                
                # 调整位移
                delta_x *= scaling_factor
                delta_y *= scaling_factor
                
                # 更新当前点的坐标
                x_list[i] = x_list[i - 1] + delta_x
                y_list[i] = y_list[i - 1] + delta_y
        # if self.v_list[idx] > 0.8:
        #     max_distance = 0.08
        #     cur_distance = np.sqrt((self.x_list[idx]-self.x_list[idx-1])**2 + (self.y_list[idx]-self.y_list[idx-1])**2)
        #     scaling_factor = max_distance / cur_distance
        #     self.x_list[idx] = self.x_list[idx-1] + (self.x_list[idx] - self.x_list[idx-1])*scaling_factor
        #     self.y_list[idx] = self.y_list[idx-1] + (self.y_list[idx] - self.y_list[idx-1])*scaling_factor
        #     self.v_list[idx] = np.sqrt((self.x_list[idx]-self.x_list[idx-1])**2 + (self.y_list[idx]-self.y_list[idx-1])**2)/0.1
        delta_x = x_list[1:] - x_list[:-1]
        delta_y = y_list[1:] - y_list[:-1]
        v_list = np.sqrt(delta_x**2 + delta_y**2) / 0.1
        v_list = np.array([v, *v_list])
        self.x_list = x_list.copy()
        self.y_list = y_list.copy()
        self.th_list = th_list.copy()
        self.v_list = v_list.copy()

        # print('control_horizon: ', self.agent_params.control_horizon, end='  ')

    def joystick_act(self):
        if self.joystick_on:
            num_cmds_per_step = self.simulator_joystick_update_ratio
            # runs through the entire planned horizon just with a cmds_step of the above
            # num_steps = int(np.floor(self.commands.k / num_cmds_per_step))
            # num_steps = int(np.floor(self.agent_params.control_horizon / num_cmds_per_step))
            num_steps = self.num_steps
            loop_num = int(num_steps/5)
            for j in range(loop_num):
                xytv_cmds = []
                for i in range(num_cmds_per_step):
                    idx = j * num_cmds_per_step + i
                    # (x, y, th, v) = self.from_conf(self.commands, idx)
                    
                    (x, y, th, v) = float(self.x_list[idx]), float(self.y_list[idx]), float(self.th_list[idx]), float(self.v_list[idx])
                    xytv_cmds.append((x, y, th, v))
                self.send_cmds(xytv_cmds, send_vel_cmds=False)

                # break if the robot finished
                if not self.joystick_on:
                    break
            
            #print('idx: ', idx, end='  ')

    def update_loop(self):
        super().pre_update()  # pre-update initialization
        self.simulator_joystick_update_ratio = int(
            np.floor(self.sim_dt / self.agent_params.dt)
        )
        while self.joystick_on:
            # gather information about the world state based off the simulator
            self.joystick_sense()
            # create a plan for the next steps of the trajectory
            self.joystick_plan()
            # send a command to the robot
            self.joystick_act()
        # complete this episode, move on to the next if need be
        self.finish_episode()
    
    def get_proper_pedestrain_data(self,current_dict):
        results = {}
        for key, value in current_dict.items():
            name = key
            x_current = value[0]
            y_current = value[1]

            if key in self.agent_trajectory_record:
                x_previous, y_previous = self.agent_trajectory_record[key]
                vx = (x_current - x_previous) / 0.1
                vy = (y_current - y_previous) / 0.1
                
            else:
                vx = 0
                vy = 0

            results[key] = {
                "name": name,
                "current_config": [
                    x_current,
                    y_current,
                    vx,
                    vy
                ]
            }

            self.agent_trajectory_record[key] = (x_current, y_current)
        return results
    
    def history_traj_library(self,current_data):
        for key, value in current_data.items():
            x, y = value[0], value[1]

            if key not in self.history:
                # If there is no history, use current data repeated 5 times
                self.history[key] = [(x, y)] * 5
            else:
                # Append current data to history
                self.history[key].append((x, y))
                # Keep only the last 5 entries
                if len(self.history[key]) > 5:
                    self.history[key].pop(0)
    
    def generate_biased_trajectories(self, x_deltas, y_deltas, ped_pos, ped_vel, num_samples, left_ratio, right_ratio):
        speed_factor = np.linalg.norm(ped_vel)

        # 判断速度是否接近零
        if speed_factor < 1e-6:
            # 当速度接近零时，设置默认的运动方向 theta = 0（朝向全局 X 轴正方向）
            theta = 0.0
        else:
            # 计算行人运动方向的角度（弧度）
            theta = np.arctan2(ped_vel[1], ped_vel[0])

        # 构建旋转矩阵（将全局坐标系旋转到行人坐标系）
        R = np.array([
            [np.cos(-theta), -np.sin(-theta)],
            [np.sin(-theta),  np.cos(-theta)]
        ])

        # 计算行人的均值轨迹（全局坐标系）
        ped_xmean = ped_pos[0] + np.arange(self.num_steps) * 0.1 * ped_vel[0]
        ped_ymean = ped_pos[1] + np.arange(self.num_steps) * 0.1 * ped_vel[1]

        # 计算轨迹的绝对位置（全局坐标系）
        x_absolute = x_deltas * speed_factor + ped_xmean
        y_absolute = y_deltas * speed_factor + ped_ymean

        # 将均值轨迹和轨迹绝对位置转换到行人坐标系
        def transform_to_pedestrian_frame(x_coords, y_coords, ped_pos):
            # x_coords 和 y_coords 的形状应为 (num_samples, plan_steps)
            coords_global = np.stack((x_coords - ped_pos[0], y_coords - ped_pos[1]), axis=-1)  # 形状 (num_samples, plan_steps, 2)
            # 应用旋转矩阵
            coords_ped_frame = np.einsum('ij,snj->sni', R, coords_global)  # 形状 (num_samples, plan_steps, 2)
            return coords_ped_frame

        # 总样本数量
        num_samples_total = x_deltas.shape[0]

        # 转换均值轨迹和轨迹绝对位置到行人坐标系
        ped_mean_coords = transform_to_pedestrian_frame(
            np.tile(ped_xmean, (num_samples_total, 1)),  # 重复均值轨迹以匹配样本数量
            np.tile(ped_ymean, (num_samples_total, 1)),
            ped_pos
        )
        traj_coords = transform_to_pedestrian_frame(
            x_absolute,
            y_absolute,
            ped_pos
        )

        # 计算轨迹相对于均值轨迹的位移向量（在行人坐标系下）
        displacement_vectors = traj_coords - ped_mean_coords  # 形状 (num_samples_total, plan_steps, 2)

        # 计算均值轨迹的切向量（在行人坐标系下）
        # 对于直线运动，切向量是恒定的，可以使用行人运动方向
        tangent_vector = np.array([1, 0])  # 在行人坐标系下，均值轨迹沿 x 轴正方向

        # 计算前三个时间步的叉乘结果
        cross_products = np.cross(
            tangent_vector,
            displacement_vectors[:, :3, :]
        )  # 形状 (num_samples_total, 3)

        # 判断前三个时间步的叉乘符号
        # 如果叉乘结果为正，表示在左侧；为负，表示在右侧
        left_mask = np.all(cross_products > 0, axis=1)
        right_mask = np.all(cross_products < 0, axis=1)

        # 获取左、右、中间轨迹的索引
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
        middle_indices = np.array([
            i for i in range(num_samples_total)
            if i not in np.concatenate((left_indices, right_indices))
        ])

        # 可用的轨迹数量
        available_left = len(left_indices)
        available_right = len(right_indices)
        available_middle = len(middle_indices)

        # 计算需要选择的轨迹数量
        num_left = int(num_samples * left_ratio)
        num_right = int(num_samples * right_ratio)
        num_middle = num_samples - num_left - num_right

        # 调整左侧轨迹数量
        if available_left == 0:
            num_middle += num_left
            num_left = 0
        elif available_left < num_left:
            extra = num_left - available_left
            num_left = available_left
            num_middle += extra

        # 调整右侧轨迹数量
        if available_right == 0:
            num_middle += num_right
            num_right = 0
        elif available_right < num_right:
            extra = num_right - available_right
            num_right = available_right
            num_middle += extra

        # 检查中间轨迹是否足够
        if available_middle < num_middle:
            extra_needed = num_middle - available_middle
            num_middle = available_middle

            # 尝试从左、右轨迹中补充
            extra_left = available_left - num_left
            extra_right = available_right - num_right

            total_extra = extra_left + extra_right

            if total_extra >= extra_needed:
                left_add = min(extra_left, extra_needed)
                num_left += left_add
                extra_needed -= left_add
                num_right += extra_needed
            else:
                # 如果仍然不足，调整总的采样数量
                num_samples = num_left + num_right + num_middle

        # 定义采样函数
        def sample_indices(indices, num_required):
            if len(indices) == 0:
                return np.array([], dtype=int)
            elif len(indices) >= num_required:
                return np.random.choice(indices, num_required, replace=False)
            else:
                # 如果数量不足，允许重复采样
                return np.random.choice(indices, num_required, replace=True)

        # 选择轨迹
        selected_left_indices = sample_indices(left_indices, num_left) if num_left > 0 else np.array([], dtype=int)
        selected_right_indices = sample_indices(right_indices, num_right) if num_right > 0 else np.array([], dtype=int)
        selected_middle_indices = sample_indices(middle_indices, num_middle) if num_middle > 0 else np.array([], dtype=int)

        # 合并选择的轨迹索引
        selected_indices = np.concatenate([selected_left_indices, selected_middle_indices, selected_right_indices])

        # 如果合并后数量超过 num_samples，截取前 num_samples 个
        selected_indices = selected_indices[:num_samples]

        # 返回最终选定的轨迹样本（全局坐标系下）
        x_sampled = x_absolute[selected_indices]
        y_sampled = y_absolute[selected_indices]

        return x_sampled, y_sampled
