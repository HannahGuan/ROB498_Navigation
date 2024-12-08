import numpy as np
import sys
sys.path.append('/home/zihanyu/SocNavBench')
# import new_brne as brne
import argparse
import matplotlib.pyplot as plt
import time
# from predictor import Predictor 
import json

def parse_args():
    parser = argparse.ArgumentParser(description="BRNE Robot Parameters")

    # Adding parameters as command line arguments
    parser.add_argument('--maximum_agents', type=int, default=8, 
                        help='Maximum number of agents BRNE will consider (including the robot)')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples assigned to each agent')
    parser.add_argument('--dt', type=float, default=0.1, 
                        help='Time step size')
    parser.add_argument('--plan_steps', type=int, default=6, 
                        help='Time steps of the planning horizon')
    parser.add_argument('--max_lin_vel', type=float, default=6, 
                        help='Maximum linear velocity allowed on the robot')
    parser.add_argument('--max_ang_vel', type=float, default=6, 
                        help='Maximum angular velocity allowed on the robot')
    parser.add_argument('--nominal_vel', type=float, default=4, 
                        help='Nominal (linear) velocity for the initial trajectory')
    parser.add_argument('--kernel_a1', type=float, default=0.2, 
                        help='Control the "straightness" of trajectory samples')
    parser.add_argument('--kernel_a2', type=float, default=0.2, 
                        help='Control the "spreadness" of trajectory samples')
    parser.add_argument('--cost_a1', type=float, default=4.0, 
                        help='Control the safety zone (more conservative if smaller)')
    parser.add_argument('--cost_a2', type=float, default=1.0, 
                        help='Control the safety zone (more conservative if larger)')
    parser.add_argument('--cost_a3', type=float, default=80.0, 
                        help='Control the safety penalty weight')
    parser.add_argument('--ped_sample_scale', type=float, default=0.1, 
                        help="Pedestrian's willingness for cooperation")
    parser.add_argument('--ad', type=float, default=-5.0, 
                        help='Aggressiveness of the optimal controller')
    parser.add_argument('--R_lin', type=float, default=1.0, 
                        help='Penalty weight on linear velocity')
    parser.add_argument('--R_ang', type=float, default=2.0, 
                        help='Penalty weight on angular velocity')
    parser.add_argument('--replan_freq', type=float, default=10.0, 
                        help='Replanning frequency in Hz')
    parser.add_argument('--people_timeout', type=float, default=5.0, 
                        help='People timeout in seconds')
    parser.add_argument('--corridor_y_min', type=float, default=-0.65, 
                        help='Lower bound of y coordinate (one side of corridor)')
    parser.add_argument('--corridor_y_max', type=float, default=0.65, 
                        help='Upper bound of y coordinate (other side of corridor)')
    parser.add_argument('--staircase_truncation', type=bool, default=False, 
                        help='Saturate F2F velocity in a staircase manner')
    parser.add_argument('--people_timeout_off', type=bool, default=True, 
                        help='Enable or disable people timeout')
    parser.add_argument('--close_stop_threshold', type=float, default=0.5, 
                        help='Threshold for safety mask leading to estop')
    parser.add_argument('--open_space_velocity', type=float, default=4, 
                        help='Nominal velocity in open space')
    parser.add_argument('--brne_activate_threshold', type=float, default=3.5, 
                        help='Distance threshold from a pedestrian to enable BRNE')

    return parser.parse_args()

class TrajTracker:
    def __init__(self, dt, max_lin_vel, max_ang_vel):
        """
        初始化 TrajTracker 实例。
        """
        self.dt = dt  # 时间步长
        self.max_lin_vel = max_lin_vel  # 最大线速度
        self.max_ang_vel = max_ang_vel  # 最大角速度

    def clamp_velocity(self, lin_vel, ang_vel):
        """
        限制线速度和角速度在允许的最大范围内。
        """
        clamped_lin_vel = np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel)
        clamped_ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
        return clamped_lin_vel, clamped_ang_vel

    def simulate_trajectory(self, initial_state, cmds):
        """
        根据初始状态和控制命令模拟轨迹，并考虑最大速度限制。
        
        参数:
        - initial_state: [x, y, theta]，初始位姿
        - cmds: 控制命令数组 (tsteps, 2)，包含线速度和角速度
        返回:
        - trajectory: 模拟的轨迹数组 (tsteps, 3)，每个时间步的 [x, y, theta]
        """
        tsteps = cmds.shape[0]
        trajectory = np.zeros((tsteps, 3))  # 存储轨迹
        state = np.array(initial_state)  # 初始化机器人状态

        for t in range(tsteps):
            lin_vel, ang_vel = cmds[t]  # 获取当前时间步的控制命令

            # 限制速度在允许的范围内
            lin_vel, ang_vel = self.clamp_velocity(lin_vel, ang_vel)

            # 更新状态
            state = self.dyn_step(state, lin_vel, ang_vel)
            trajectory[t] = state.copy()  # 记录当前状态

        return trajectory

    def dyn_step(self, state, lin_vel, ang_vel):
        """
        根据给定的线速度和角速度更新机器人的状态。
        
        参数:
        - state: 当前位姿 [x, y, theta]
        - lin_vel: 当前线速度
        - ang_vel: 当前角速度
        返回:
        - 更新后的位姿 [x, y, theta]
        """
        x, y, theta = state
        new_x = x + lin_vel * np.cos(theta) * self.dt
        new_y = y + lin_vel * np.sin(theta) * self.dt
        new_theta = theta + ang_vel * self.dt
        return np.array([new_x, new_y, new_theta])


class brne_prediction():
    def __init__(self) -> None:
        self.num_agents = 8
        self.num_samples = 196
        self.dt = 0.1
        self.plan_steps = 10
        self.max_lin_vel = 1.2
        self.max_ang_vel = 1
        self.nominal_vel = 1
        self.kernel_a1 = 0.2
        self.kernel_a2 = 0.2
        self.cost_a1 = 4.0
        self.cost_a2 = 1.0
        self.cost_a3 = 80.0
        self.ped_sample_scale = 0.1
        self.replan_freq = 20.0
        # self.people_timeout = Duration(seconds=args.people_timeout)
        self.corridor_y_min = -0.65
        self.corridor_y_max = 0.65
        self.staircase_truncation = False
        self.people_timeout_off = True
        self.close_stop_threshold = 0.5
        self.open_space_velocity = 1.2
        self.brne_activate_threshold = 3.5
        self.dorminate_ratio = 0.8
        self.part_ratio = 0.2

        self.x_opt_trajs = np.zeros((self.num_agents, self.plan_steps))  # optimal trajectories from BRNE
        self.y_opt_trajs = np.zeros((self.num_agents, self.plan_steps))
        self.robot_pose = np.zeros(3)  # the robot's initial pose
        self.robot_goal = None   # the robot's goal
        self.robot_traj = []
        # initialize the BRNE covariance matrix here
        tlist = np.arange(self.plan_steps) * self.dt
        train_ts = np.array([tlist[0]])
        train_noise = np.array([1e-04])
        test_ts = tlist

        self.cov_Lmat, self.cov_mat = brne.get_Lmat_nb(train_ts, test_ts, train_noise, self.kernel_a1, self.kernel_a2)
        self.curr_ped_array = [] # 存储当前时间步检测到的所有行人位置。
        self.prev_ped_array = [] # 存储上一时间步的行人位置。
 
        self.close_stop_flag = False # 判断机器人是否需要紧急停止。

        self.brne_first_time = True
        self.cmd_tracker = TrajTracker(dt=self.dt, max_lin_vel=self.max_lin_vel, max_ang_vel=self.max_ang_vel)

    
    def brne_cb(self, ped_data, robot_goal, robot_pose,tool):
        start_time = time.time()

        ped_info_list = []
        dists2peds = []
        # now = self.get_clock().now()
        # if self.people_timeout_off == False:
        #     for ped_ident, (ped, stamp) in list(self.ped_msg_buffer.items()):
        #         if now - stamp > self.people_timeout:
        #             del self.ped_msg_buffer[ped_ident]        # 开始计时并处理超时行人数据

        # we go through each perceived pedestrian and save the information
        for ped in ped_data.keys():
            # ped_data: {"prerec_0000":{"name":'prerec_0000', 'current_config':[x,y,vx,vy]},}
            dist2ped = np.sqrt((robot_pose[0]-ped_data[ped]['current_config'][0])**2 + (robot_pose[1]-ped_data[ped]['current_config'][1])**2)
            if dist2ped < self.brne_activate_threshold:  # only consider pedestrians within the activate threshold
                ped_info = np.array([
                    ped_data[ped]['current_config'][0], ped_data[ped]['current_config'][1], ped_data[ped]['current_config'][2], ped_data[ped]['current_config'][3],int(ped[-1])
                ])
                ped_info_list.append(ped_info)

                dists2peds.append(dist2ped)    # 计算机器人与行人之间的距离，并将距离小于 brne_activate_threshold 的行人纳入考虑范围。

        ped_info_list = np.array(ped_info_list)
        self.num_peds = len(ped_info_list)

        dists2peds = np.array(dists2peds)

        # compute how many pedestrians we are actually interacting with
        num_agents = np.minimum(self.num_peds+1, self.num_agents) # 实际参与路径规划的agent数量（包括机器人和行人）。

        if num_agents > 1:
            ped_indices = np.argsort(dists2peds)[:num_agents-1]  # we only pick the N closest pedestrian to interact with
            robot_state = robot_pose.copy()
#########################################################################################
            x_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples*2, self.plan_steps, self.cov_Lmat) # 使用 多元正态分布生成轨迹样本 (x_pts 和 y_pts)，为每个行人生成多个可能的路径。
            y_pts = brne.mvn_sample_normal((num_agents-1) * self.num_samples*2, self.plan_steps, self.cov_Lmat)

            # self.get_logger().info(f'X and y pts shape {x_pts.shape} {y_pts.shape}')
            # ctrl space configuration here
            xtraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))
            ytraj_samples = np.zeros((
                num_agents * self.num_samples, self.plan_steps
            ))

            closest_dist2ped = 100.0
            closest_ped_pos = np.zeros(2) + 100.0 #用一个很大的值初始化距离最近的行人
            result_per_frame = {}
            for i, ped_id in enumerate(ped_indices): # 两个工作，1.对每个行人生成x,y方向的样本，基于设定的参数和constant speed model的预测，生成行人轨迹均值。2.更新最近的行人位置
                ped_pos = ped_info_list[ped_id][:2]
                ped_vel = ped_info_list[ped_id][2:4]
                idx = ped_info_list[ped_id][-1]
                agent_name = list(ped_data.keys())[int(idx)]
                speed_factor = np.linalg.norm(ped_vel)
                # ped_xmean = ped_pos[0] + np.arange(self.plan_steps) * self.dt * ped_vel[0] # 使用行人当前的位置和速度，计算未来各个时间步的 X 和 Y 坐标的均值。
                # ped_ymean = ped_pos[1] + np.arange(self.plan_steps) * self.dt * ped_vel[1]
                
                ped_xmean, ped_ymean = tool.start_prediction(agent_name, ped_data)
                
                xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    x_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_xmean # 生成的轨迹样本 = （随机轨迹样本 × 速度缩放）+ 预测位置的均值。
                ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] = \
                    y_pts[i*self.num_samples : (i+1)*self.num_samples] * speed_factor + ped_ymean # 缩放和平移样本，确保轨迹符合行人的速度和未来位置预测。

                dist2ped = np.linalg.norm([
                    robot_state[:2] - ped_pos[:2]
                ])
                if dist2ped < closest_dist2ped:
                    closest_dist2ped = dist2ped
                    closest_ped_pos = ped_pos.copy() # 更新 最近行人的距离 和 最近行人的位置。

            st = robot_state.copy()

            if robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                goal = robot_goal[:2]

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)  # 根据机器人的朝向（θ），计算一个与其垂直的向量 axis_vec。
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal # axis_vec在目标向量上的投影长度，用于估计运动曲线的半径。
            radius = 0.5 * dist2goal / proj_len  # 基于距离和投影计算出机器人转弯的曲率半径。

            if st[2] > 0.0:
                ut = np.array([self.nominal_vel, -self.nominal_vel/radius]) # 根据机器人的朝向，计算 线速度和角速度。
            else:
                ut = np.array([self.nominal_vel, self.nominal_vel/radius])
###########################################
# 可以修改掉，直接对轨迹采样，无需对控制命令采样
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1)) # nominal_cmds：将控制命令复制 plan_steps 次，生成时间序列控制指令。
            # self.get_logger().info(f"Nominal commands {nominal_cmds.shape}\n{nominal_cmds}")
            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, self.max_lin_vel, self.max_ang_vel, self.num_samples # 基于当前的控制指令采样一系列控制指令
            )
            # self.get_logger().info(f"ulist {ulist_essemble.shape}\n{ulist_essemble}")
            #tiles = np.tile(robot_state, reps=(self.num_samples,1)).T
            # self.get_logger().info(f"Tiles {tiles.shape}\n{tiles}")
            traj_essemble = brne.traj_sim_essemble(                                # 这段代码利用控制命令集生成多条轨迹
                                                                                   # traj_essemble.shape = (tsteps, 3, num_samples)
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )
            # self.get_logger().info(f"traj {traj_essemble.shape}\n{traj_essemble}")
            # self.get_logger().info(f"for xtraj samples {(traj_essemble[:,0,:].T).shape}\n{traj_essemble[:,0,:].T}")
            xtraj_samples[0:self.num_samples] = traj_essemble[:,0,:].T           # 将轨迹样本的 X 和 Y 方向分别存入 xtraj_samples 和 ytraj_samples
            ytraj_samples[0:self.num_samples] = traj_essemble[:,1,:].T
######################
######################
            # generate sample weight mask for the closest pedestrian
            robot_xtrajs = traj_essemble[:,0,:].T
            robot_ytrajs = traj_essemble[:,1,:].T
            robot_samples2ped = (robot_xtrajs - closest_ped_pos[0])**2 + (robot_ytrajs - closest_ped_pos[1])**2
            robot_samples2ped = np.min(np.sqrt(robot_samples2ped), axis=1)
            safety_mask = (robot_samples2ped > self.close_stop_threshold).astype(float) # 计算哪些样本的距离和最近行人的距离远，生成mask
            # self.get_logger().info(f'safety mask\n{safety_mask}')
            safety_samples_percent = safety_mask.mean() * 100
            # self.get_logger().debug('percent of safe samples: {:.2f}%'.format(safety_samples_percent))
            # self.get_logger().debug('dist 2 ped: {:.2f} m'.format(closest_dist2ped))


            self.close_stop_flag = False
            if np.max(safety_mask) == 0.0:
                safety_mask = np.ones_like(safety_mask)
                self.close_stop_flag = True
            # self.get_logger().debug('safety mask: {}'.format(safety_mask))

            # BRNE OPTIMIZATION HERE !!!
            weights = brne.brne_nav(
                xtraj_samples, ytraj_samples,
                num_agents, self.plan_steps, self.num_samples,
                self.cost_a1, self.cost_a2, self.cost_a3, self.ped_sample_scale,
                self.corridor_y_min, self.corridor_y_max, goal[0], goal[1]
            ) # 用BRNE方法生成权重矩阵

            # self.get_logger().info(f"Weights\n{weights}")

            if self.brne_first_time:
                print("BRNE initialization complete!")
                self.brne_first_time = False

            if weights is None:
                print("We are going out of bounds. Stop going to this goal")
                return

            # apply safety mask
            weights[0] *= safety_mask
            if (np.mean(weights[0]) != 0):
                weights[0] /= np.mean(weights[0])
            else:
                print("Stopping because of safety mask")

##################
#此处可以改，直接去乘轨迹，不用转向crtl cmds

            # generate optimal ctrl cmds and update buffer
            opt_cmds_1 = np.mean(ulist_essemble[:,:,0] * weights[0], axis=1)
            opt_cmds_2 = np.mean(ulist_essemble[:,:,1] * weights[0], axis=1)
            # self.get_logger().info(f"opt cmds 1 {opt_cmds_1}")
            self.cmds = np.array([opt_cmds_1, opt_cmds_2]).T
##################
#此处可以改，不用crtl cmds转为traj
            self.cmds_traj = self.cmd_tracker.simulate_trajectory(robot_state, self.cmds)

            ped_trajs_x = np.zeros((num_agents-1, self.plan_steps))
            ped_trajs_y = np.zeros((num_agents-1, self.plan_steps))
            for i in range(num_agents-1):
                ped_trajs_x[i] = \
                    np.mean(xtraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
                ped_trajs_y[i] = \
                    np.mean(ytraj_samples[(i+1)*self.num_samples : (i+2)*self.num_samples] * weights[i+1][:,np.newaxis], axis=0)
##################
#找到返回值是什么

            if robot_goal is None or self.close_stop_flag == True:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))
                

            # for smoothness, we allow the robot to execute the first 5 time steps from the buffer
            # if self.cmd_counter > 0:
            #     self.cmd_counter = 0


        else:  # if no pedestrian around, go straight to the goal
            self.close_stop_flag = False
            robot_state = robot_pose.copy()
            st = robot_state.copy()

            if robot_goal is None:
                goal = np.array([6.0, 0.0])
            else:
                goal = robot_goal[:2]

            if st[2] > 0.0:
                theta_a = st[2] - np.pi/2
            else:
                theta_a = st[2] + np.pi/2
            axis_vec = np.array([
                np.cos(theta_a), np.sin(theta_a)
            ])
            vec2goal = goal - st[:2]
            dist2goal = np.linalg.norm(vec2goal)
            proj_len = (axis_vec @ vec2goal) / (vec2goal @ vec2goal) * dist2goal
            radius = 0.5 * dist2goal / proj_len

            nominal_vel = self.open_space_velocity
            if st[2] > 0.0:
                ut = np.array([nominal_vel, -nominal_vel/radius])
            else:
                ut = np.array([nominal_vel,  nominal_vel/radius])
            nominal_cmds = np.tile(ut, reps=(self.plan_steps,1))

##################
#此处可以改，不用crtl cmds转为traj
            ulist_essemble = brne.get_ulist_essemble(
                nominal_cmds, nominal_vel+0.05, self.max_ang_vel, self.num_samples
            )
            traj_essemble = brne.traj_sim_essemble(
                np.tile(robot_state, reps=(self.num_samples,1)).T,
                ulist_essemble,
                self.dt
            )
            end_pose_essemble = traj_essemble[-1, 0:2, :].T
            dists2goal_essemble = np.linalg.norm(end_pose_essemble - goal, axis=1)
            opt_cmds = ulist_essemble[:, np.argmin(dists2goal_essemble), :]
##################
#此处可以改，不用crtl cmds转为traj
            self.cmds = opt_cmds
            self.cmds_traj = self.cmd_tracker.simulate_trajectory(robot_state, self.cmds)

            # if self.cmd_counter > 0:
            #     self.cmd_counter = 0
##################
#找到返回值是什么
            # self.publish_trajectory(self.opt_traj_pub, self.cmds_traj[:,0], self.cmds_traj[:,1])
            # self.publish_markers([], [])

            if robot_goal is None:
                self.cmds = np.zeros((self.plan_steps, 2))
                self.cmds_traj = np.tile(robot_state, reps=(self.plan_steps,1))
            

        end_time = time.time()
        diff = end_time - start_time
        # diff_sec = diff.sec + diff.nanosec*1e-9
        # self.get_logger().debug(f"Agents: {num_agents} Timer: {diff_sec}")
        return self.cmds_traj



if __name__ == "__main__":
    args = parse_args()
    # agent_data = {'prerec_0000': {'name': 'prerec_0000', 'start_config': [11.051899909973145, 5.9962158203125], 'goal_config': [17.653799057006836, 6.98838996887207, 0.02292591892182827]}, 
    #               'prerec_0001': {'name': 'prerec_0001', 'start_config': [13.589799880981445, 7.168900012969971], 'goal_config': [17.656999588012695, 7.105800151824951, -0.04700012505054474]}, 
    #               'prerec_0002': {'name': 'prerec_0002', 'start_config': [7.753699779510498, 3.9261999130249023], 'goal_config': [17.424699783325195, 10.365799903869629, 0.25944942235946655]}, 
    #               'prerec_0003': {'name': 'prerec_0003', 'start_config': [3.648699998855591, 5.9633049964904785], 'goal_config': [17.86680030822754, 8.049099922180176, 0.17759616672992706]}, 
    #               'prerec_0004': {'name': 'prerec_0004', 'start_config': [16.12380027770996, 6.523139953613281], 'goal_config': [2.620300054550171, 12.040200233459473, 2.677945137023926]}, 
    #               'prerec_0005': {'name': 'prerec_0005', 'start_config': [16.584800720214844, 5.732049942016602], 'goal_config': [14.953200340270996, 5.753290176391602, -2.304856300354004]}, 
    #               'prerec_0006': {'name': 'prerec_0006', 'start_config': [16.007299423217773, 3.118299961090088], 'goal_config': [1.246399998664856, 6.3774800300598145, 2.898289918899536]}, 
    #               'prerec_0007': {'name': 'prerec_0007', 'start_config': [16.485599517822266, 3.7939999103546143], 'goal_config': [1.1993999481201172, 7.178100109100342, 3.135693073272705]}}

    # steps = 30
    # ped_data_list = []

    # for step in range(steps):
    #     step_data = {}
    #     for key, data in agent_data.items():
    #         start = np.array(data['start_config'])
    #         goal = np.array(data['goal_config'][:2])  

    #         x = np.linspace(start[0], goal[0], steps)[step]
    #         y = np.linspace(start[1], goal[1], steps)[step]

    #         if step == 0:
    #             vx, vy = 0, 0
    #         else:
    #             x_prev = np.linspace(start[0], goal[0], steps)[step-1]
    #             y_prev = np.linspace(start[1], goal[1], steps)[step-1]
    #             vx = (x-x_prev)/0.1
    #             vy = (y-y_prev)/0.1

    #         step_data[key] = {
    #             'name': data['name'],
    #             'current_config': [float(x), float(y), float(vx), float(vy)]
    #         }

    #     ped_data_list.append(step_data)

    with open('/home/zihanyu/SocNavBench/tests/pedestrian_data.json', 'r') as file:
        ped_data_list = json.load(file)
    
    tool = Predictor(ped_data_list[0])
    brne_function = brne_prediction(args)

    robot_pose =  [12.5, 2.5, 1.7]
    #robot_pose =  [13, 2.5, 1.7]
    robot_goal = [12, 9, 0]
    robot_trajectory =[]

    for i, pedestrian_data in enumerate(ped_data_list):
        robot_trajectory.append(robot_pose[:2])
        if i == 0:
            continue
        t1 = time.time()
        next_pose = brne_function.brne_cb(pedestrian_data, np.array(robot_goal), np.array(robot_pose))
        t2 = time.time()
        print(t2-t1)
        ped_x = []
        ped_y = []
        for agent_id, agent_info in pedestrian_data.items():
            config = agent_info['current_config']
            ped_x.append(config[0])  # x 坐标
            ped_y.append(config[1])  # y 坐标

        # 创建绘图
        plt.figure(figsize=(10, 8))

        # 绘制行人位置
        plt.scatter(ped_x, ped_y, c='blue', label='Pedestrians', alpha=0.6, edgecolors='w', s=100)

        # 绘制 next_pose 位置
        plt.scatter(robot_pose[0], robot_pose[1], c='green',  label='Robbot Pose', s=100)

        plt.scatter(robot_goal[0], robot_goal[1], c='red', marker='X', label='Robbot Goal', s=200)

        robot_x = [pos[0] for pos in robot_trajectory]
        robot_y = [pos[1] for pos in robot_trajectory]

        # 绘制机器人轨迹
        plt.plot(robot_x, robot_y, c='purple', label='Robot Trajectory')

        # 设置图表标题和标签
        plt.title('Visualization')
        plt.xlabel('X ')
        plt.ylabel('Y')

        # 显示图例
        plt.legend()

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.savefig('/home/zihanyu/SocNavBench/image2/'+str(i)+'.jpg')

        robot_pose = next_pose[0]
    final_pose = robot_pose
    # ped_data: {"prerec_0000":{"name":'prerec_0000', 'current_config':[x,y,vx,vy]},}