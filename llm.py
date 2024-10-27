from openai import OpenAI
client = OpenAI()
import time

# the direction information is contained in the history trajectory.
system_message = """
Role: You are a walking pedestrian. You need to follow your history trajectory, current state and velocity to walk, keep a safety distance with other pedestrians and the robot.
Objective:
- Generate your walking intention in the next 1 second.
Inputs:
1. Surrounding Information: Info about surrounding pedestrians and the robot, including positions and velocities (in the format of [x,y,vx,vy]).
2. Your current State: Your current state including position and velocity (in the format of [x,y,vx,vy]).
3. History trajectory: Your history trajectory in the past 5 frames, given by 5 discrete waypoints.
Output:
- Only output the word described your walking intention in the next 1 second (Select one word from left, right and straight).
"""

example_user_message = '''
Input: 
Surrounding Information: pedestrian 1: [4,10,1,5]. 
Your current state: [3,9,1,5]. 
History trajectory: [(0,0),(0,0),(0,0),(1,1),(2,4)]
'''

user_message  = f"\n"
user_message += f"Input: Surrounding Information:\n"


t1 = time.time()
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": user_message
        }
    ]
)

print(completion.choices[0].message.content)
t2 = time.time()
print(t2-t1)