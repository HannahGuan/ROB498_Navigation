import numpy as np
from openai import OpenAI
# client = OpenAI(api_key="sk-proj-sssOcRxqLs1OgMXmphGKT3BlbkFJGTJS2jCCJMoHednhstwh")
client = OpenAI(api_key="")
import time
import math

class Predictor():
    """
    Wrap everything into an API
    Main function: 
    """

    def __init__(self, first_frame = None, num_ped = 8):
        example_user_message = '''
            Input: 
            Surrounding Information: {pedestrian 1: [4,10,1,5]; pedestrain 2: [2,4,1,3]}. 
            Your current state: [3,9,1,5]. 
            History trajectory: [(0,0),(0,0),(0,0),(1,1),(2,4)]
        '''
        self.template_one = '''
            Input: 
            Surrounding Information: {nearby}. 
            Your current state: {current_state}. 
            History trajectory: {history}
        '''
        self.system_message = """
            Role: You are a walking pedestrian. You are walking in a hotel hall. You need to follow your history trajectory, current state and velocity to walk, keep a safety distance with other pedestrians and the robot.
            Objective:
            - Generate your walking intention in the next 1 second.
            Inputs:
            1. Surrounding Information: Info about surrounding pedestrians and the robot, including positions and velocities (in the format of [x,y,vx,vy]).
            2. Your current State: Your current state including position and velocity (in the format of [x,y,vx,vy]).
            3. History trajectory: Your history trajectory in the past 5 frames, given by 5 discrete waypoints.
            Output:
            - Only output the word described your walking intention in the next 1 second (Select one word from left, right and straight).

            ##OUTPUT FORMAT
            left OR right OR straight

        """
        self.radius = 3.5
        if first_frame:
            self.saved_trajectory = self.initialize_saved_trajectory(first_frame)
    
    def initialize_saved_trajectory(self, first_frame_data):
        saved_trajectory = {
            key: [] for key in first_frame_data.keys()
        }

        for key, value in first_frame_data.items():
            x, y = value["current_config"][:2]
            saved_trajectory[key] = [(x, y)] * 5

        return saved_trajectory

    def reset(self):
        """
        Reset the saved data & distance to use
        """
        self.surrounding_distance = 3.5
        self.saved_trajectory = []


    def set_radius(self, new_dist):
        self.radius = new_dist

    def set_history_library(self,history):
        self.history_library = history


    def make_prediction(self, new_timeData, robot_pose, seeTime = False, seePrompt = False):
        """
        Main function to use! Take in the all pedastrains's location at a time and return a list of predicted intentions 
        """
        prediction_map = {f"prerec_{str(i).zfill(4)}": [] for i in range(len(new_timeData))}
        for agent_name, agent_data in new_timeData.items():
            current_location = agent_data["current_config"]
            # Calculate surrounding information
            surrounding_info = []
            pedestrian_count = 0
            for other_name, other_data in new_timeData.items():
                if other_name != agent_name:
                    other_location = other_data["current_config"]
                    distance = math.sqrt((current_location[0] - other_location[0]) ** 2 + 
                                        (current_location[1] - other_location[1]) ** 2)
                    if distance <= self.radius:
                        surrounding_info.append(f"pedestrian {pedestrian_count}: {other_location}")
                        pedestrian_count += 1
            if math.sqrt((current_location[0] - robot_pose[0]) ** 2 + 
                                        (current_location[1] - other_location[1]) ** 2) <= self.radius:
                surrounding_info.append(f"robot: {robot_pose}")
            # Format surrounding information as specified
            surrounding_info_text = "{ " + "; ".join(surrounding_info) + " }" if surrounding_info else "None"
            
            # Fill in the prompt template
            prediction = self.make_onePred(surrounding_info_text, current_location, self.saved_trajectory[agent_name], seeTime, seePrompt)
            prediction = prediction.lower()
            if prediction not in ['left', 'right', 'straight']:
                print("error in making prediction for ", agent_name, "; wrong prediction format: ", prediction)

            # Update saved trajectory
            if len(self.saved_trajectory[agent_name]) >= 5:
                self.saved_trajectory[agent_name].pop(0)  # Keep only the last five points
            self.saved_trajectory[agent_name].append(tuple(current_location[:2]))
            prediction_map[agent_name] = prediction
        return prediction_map
    
    def update_saved_trajectory(self,new_timeData):
        for agent_name, agent_data in new_timeData.items():
            current_location = agent_data["current_config"]
            if len(self.saved_trajectory[agent_name]) >= 5:
                self.saved_trajectory[agent_name].pop(0)  # Keep only the last five points
            self.saved_trajectory[agent_name].append(tuple(current_location[:2]))
        
    def get_one_prediction_v1(self, agent_name, new_timeData, robot_pose, seeTime = False, seePrompt = False):
        self.update_saved_trajectory(new_timeData)
        current_location = new_timeData[agent_name]["current_config"]
        surrounding_info = []
        pedestrian_count = 0
        for other_name, other_data in new_timeData.items():
            if other_name != agent_name:
                other_location = other_data["current_config"]
                distance = math.sqrt((current_location[0] - other_location[0]) ** 2 + 
                                    (current_location[1] - other_location[1]) ** 2)
                if distance <= self.radius:
                    surrounding_info.append(f"pedestrian {pedestrian_count}: {other_location}")
                    pedestrian_count += 1
        if math.sqrt((current_location[0] - robot_pose[0]) ** 2 + 
                                        (current_location[1] - other_location[1]) ** 2) <= self.radius:
                surrounding_info.append(f"robot: {robot_pose}")
        surrounding_info_text = "{ " + "; ".join(surrounding_info) + " }" if surrounding_info else "None"

        prediction = self.make_onePred(surrounding_info_text, current_location, self.saved_trajectory[agent_name], seeTime, seePrompt)
        
        if prediction not in ['left', 'right', 'straight']:
                print("error in making prediction for ", agent_name, "; wrong prediction format: ", prediction)

        return prediction
    
    def get_one_prediction(self, agent_name, new_timeData, robot_pose, seeTime = False, seePrompt = False):
        current_location = new_timeData[agent_name]["current_config"]
        surrounding_info = []
        pedestrian_count = 0
        for other_name, other_data in new_timeData.items():
            if other_name != agent_name:
                other_location = other_data["current_config"]
                distance = math.sqrt((current_location[0] - other_location[0]) ** 2 + 
                                    (current_location[1] - other_location[1]) ** 2)
                if distance <= self.radius:
                    surrounding_info.append(f"pedestrian {pedestrian_count}: {other_location}")
                    pedestrian_count += 1
        if math.sqrt((current_location[0] - robot_pose[0]) ** 2 + 
                                        (current_location[1] - robot_pose[1]) ** 2) <= self.radius:
                surrounding_info.append(f"robot: {robot_pose}")
        surrounding_info_text = "{ " + "; ".join(surrounding_info) + " }" if surrounding_info else "None"

        prediction, result = self.make_onePred(surrounding_info_text, current_location, self.history_library[agent_name], seeTime, seePrompt)
        
        if prediction not in ['left', 'right', 'straight']:
                print("error in making prediction for ", agent_name, "; wrong prediction format: ", prediction)

        return prediction, result

    def make_onePred(self, nearby, current_state, history, seeTime, seePrompt):
        formatted_text = self.template_one.replace("{nearby}", nearby)\
                                  .replace("{current_state}", str(current_state))\
                                  .replace("{history}", str(history))
        t1 = time.time()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": formatted_text
                }
            ],
            logprobs=True,
            top_logprobs=3,
        )
        t2 = time.time()
        if seeTime:
            print('this call takes the time: ', t2-t1)
        if seePrompt:
            print(formatted_text)
        top_three_logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        result = {}
        for i, logprob in enumerate(top_three_logprobs, start=1):
            result[logprob.token]=np.round(np.exp(logprob.logprob)*100,2)/100
            
        return completion.choices[0].message.content, result


if __name__=="__main__":
    import json

    with open('/home/zihanyu/SocNavBench/tests/pedestrian_data.json', 'r') as file:
        data = json.load(file)

    tool = Predictor(data[0])
    robot_pose =  [8, 5, 0]
    for i in range(len(data)):
        tool.make_prediction(data[i],robot_pose, seeTime=True, seePrompt=True)