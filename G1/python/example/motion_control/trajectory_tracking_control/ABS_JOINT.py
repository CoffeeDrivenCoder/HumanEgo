from a2d_sdk.robot import RobotController,RobotDds

# 初始化机器人控制器
robot_controller = RobotController()
robot = RobotDds()
def execute(robot, robot_controller,action):
        robot_states = {}
        robot_actions = []
        robot_link = "base_link"
        trajectory_ref_time = 1.0
        wait_control_time = 0.5
        infer_timestamp = action["observation_timestamp"]
        robot_states["head"] = action["head_joint_states"]
        robot_states["waist"] = action["waist_joint_states"]
        robot_states["arm"] = action["arm_joint_states"]

        arm_joint_action = action["arm_cmd"]
        for i in range(len(arm_joint_action)):
            robot_action = {
                "left_arm": {
                    "action_data": arm_joint_action[i][:7],
                    "control_type": "ABS_JOINT"
                },
                "right_arm": {
                    "action_data": arm_joint_action[i][7:14],
                    "control_type": "ABS_JOINT"
                }
            }
            robot_actions.append(robot_action)
        time.sleep(1)
        robot_controller.trajectory_tracking_control(infer_timestamp, 
                                                      robot_states, 
                                                      robot_actions, 
                                                      robot_link, 
                                                      trajectory_ref_time)
        time.sleep(wait_control_time)
        print(f"Infer timestamp: {infer_timestamp}")
        print(f"Robot states: {robot_states}")
        print(f"Robot actions: {robot_actions}")
        while True:
            state, _= robot.arm_joint_states()
            print(f"Arm joint states: {state}")

if __name__ == "__main__":
    import time
    action = {
        "observation_timestamp": int(time.time() * 1e9),  # 当前时间戳（纳秒）
        "head_joint_states": [0.0, 0.0],
        "waist_joint_states": [0.5, 0.3],
        "arm_joint_states": [0] * 14,
        "arm_cmd":[[0.1]*14],
        
    }

    execute(robot,robot_controller, action)
    print("Done")