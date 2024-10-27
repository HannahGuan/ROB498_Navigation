import numpy as np
import time
import matplotlib.pyplot as plt

agent_data = {'prerec_0000': {'name': 'prerec_0000', 'start_config': [11.051899909973145, 5.9962158203125], 'goal_config': [17.653799057006836, 6.98838996887207]}, 
                  'prerec_0001': {'name': 'prerec_0001', 'start_config': [13.589799880981445, 7.168900012969971], 'goal_config': [17.656999588012695, 7.105800151824951]}, 
                  'prerec_0002': {'name': 'prerec_0002', 'start_config': [7.753699779510498, 3.9261999130249023], 'goal_config': [17.424699783325195, 10.365799903869629]}, 
                  'prerec_0003': {'name': 'prerec_0003', 'start_config': [3.648699998855591, 5.9633049964904785], 'goal_config': [17.86680030822754, 8.049099922180176]}, 
                  'prerec_0004': {'name': 'prerec_0004', 'start_config': [16.12380027770996, 6.523139953613281], 'goal_config': [2.620300054550171, 12.040200233459473]}, 
                  'prerec_0005': {'name': 'prerec_0005', 'start_config': [16.584800720214844, 5.732049942016602], 'goal_config': [14.953200340270996, 5.753290176391602]}, 
                  'prerec_0006': {'name': 'prerec_0006', 'start_config': [16.007299423217773, 3.118299961090088], 'goal_config': [1.246399998664856, 6.3774800300598145]}, 
                  'prerec_0007': {'name': 'prerec_0007', 'start_config': [16.485599517822266, 3.7939999103546143], 'goal_config': [1.1993999481201172, 7.178100109100342]}}

steps = 30
ped_data_list = []

for step in range(steps):
    step_data = {}
    for key, data in agent_data.items():
        start = np.array(data['start_config'])
        goal = np.array(data['goal_config'][:2])  

        x = np.linspace(start[0], goal[0], steps)[step]
        y = np.linspace(start[1], goal[1], steps)[step]

        if step == 0:
            vx, vy = 0, 0
        else:
            x_prev = np.linspace(start[0], goal[0], steps)[step-1]
            y_prev = np.linspace(start[1], goal[1], steps)[step-1]
            vx = (x-x_prev)/0.1
            vy = (y-y_prev)/0.1

        step_data[key] = {
            'name': data['name'],
            'current_config': [float(x), float(y), float(vx), float(vy)]
        }

    ped_data_list.append(step_data)


robot_pose =  [16.0073, 3.1183]
robot_goal = [17.4247, 10.3658]

ped_data_list.to_csv(ped_data_list)


brne_function = brne_reproduce(args) 

for i, pedestrian_data in enumerate(ped_data_list):
    t1 = time.time()
    next_pose = brne_function.brne_cb(pedestrian_data, np.array(robot_goal), np.array(robot_pose))
###########################
    '''
    Here for the original brne, they only need the position and vx vy of each agent. 

    But for our method, when the pedestrian is within a threshhold (3.5m), then this pedesdtrian is classified as a dangerous pedestrian.
    For every dangerous pedestrian, we need to use the LLM to capture its intention. 
    The information we needed:
    1. current position and velocity of this pedestrian, store it as [x,y,vx,vy]
    2. history trajectory (5 frames) for this pedestrian, store it as [(x,y),(,),(,),(,),(,)]
    3. dangerous agents for this pedestrian, store it as {id1:[x,y,vx,vy],id2:[x,y,vx,vy]}

    After getting this information, we form the prompt to input into the LLM.
    '''
###########################
    t2 = time.time()
    print(t2-t1)
    ped_x = []
    ped_y = []
    for agent_id, agent_info in pedestrian_data.items():
        config = agent_info['current_config']
        ped_x.append(config[0])  # x 
        ped_y.append(config[1])  # y 

    plt.figure(figsize=(10, 8))

    plt.scatter(ped_x, ped_y, c='blue', label='Pedestrians', alpha=0.6, edgecolors='w', s=100)

    plt.scatter(robot_pose[0], robot_pose[1], c='green',  label='Robbot Pose', s=100)

    plt.scatter(robot_goal[0], robot_goal[1], c='red', marker='X', label='Robbot Goal', s=200)

    plt.title('Visualization')
    plt.xlabel('X ')
    plt.ylabel('Y')

    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig('/home/zihanyu/SocNavBench/image/'+str(i)+'.jpg')
    robot_pose = next_pose[0]
"""