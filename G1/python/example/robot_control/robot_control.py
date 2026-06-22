import time
import threading
import numpy as np
import argparse

from a2d_sdk.robot import RobotDds as Robot

def robot_control_task(robot : Robot, stop_event):
    while not stop_event.is_set():
        if len(robot._head_joint_state.position) == 0 or len(robot._waist_joint_state.motor_states) == 0:
            print("Head or Waist state is empty, wait for 1 more second...")
            time.sleep(1)
            continue
        else:
            break
    if len(robot._head_joint_state.position) == 0 or len(robot._waist_joint_state.motor_states) == 0:
        print("State is not ready, reset failed")
        return

    print("Reset robot")
    robot.reset(arm_positions=robot.arm_initial_joint_position, head_positions=[0.0, 0.0], \
                 waist_positions=[0.0, 0.0])

    print("Move gripper from 0.0 to 1.0")
    for i in np.linspace(0.0, 1.0, 10):
        robot.move_gripper([i, i])
        time.sleep(1)
    print("Move gripper from 1.0 to 0.0")
    for i in np.linspace(1.0, 0.0, 10):
        robot.move_gripper([i, i])
        time.sleep(1)

    print("Move wheel")
    round = 0
    while round < 50 and not stop_event.is_set():
        robot.move_wheel(0.1, 0.0)
        time.sleep(0.1)
        round += 1
    round = 0
    while round < 50 and not stop_event.is_set():
        robot.move_wheel(-0.1, 0.0)
        time.sleep(0.1)
        round += 1

    print("Move head")
    init_head_state = robot._body_pose_joint_states[:2]
    init_waist_state = robot._body_pose_joint_states[2:4]
    # robot.move_wheel(0.1, 0.0)
    head_state = [n - 10 for n in init_head_state]
    print(f"Move Head State: {head_state}, {type(head_state[0])}")
    robot.move_head(head_state)

    round = 0
    while round < 10 and not stop_event.is_set():
        init_head_state = robot._body_pose_joint_states[:2]
        init_waist_state = robot._body_pose_joint_states[2:4]
        print(f"Init Head State: {init_head_state}")
        print(f"Init Waist State: {init_waist_state}")

        if round % 4 == 0:
            print("Move head to left and gripper to 1.0")
            # add 0.02 to each element of init_head_state
            head_state = [n + 10 for n in init_head_state]
            print(f"Move Head State: {head_state}, {type(head_state[0])}")
            robot.move_head(head_state)
            # robot.move_waist([n + 0.02 for n in init_waist_state])
            robot.move_gripper([1.0, 1.0])
        elif round % 4 == 1:
            print("Move head to right and gripper to 0.7")
            head_state = [n - 10 for n in init_head_state]
            print(f"Move Head State: {head_state}, {type(head_state[0])}")
            robot.move_head(head_state)
            robot.move_gripper([0.7, 0.7])
        elif round % 4 == 2:
            print("Move waist to up and gripper to 0.25")
            waist_state = [n + 5 for n in init_waist_state]
            print(f"Move Waist State: {waist_state}, {type(waist_state[0])}")
            robot.move_waist(waist_state)
            robot.move_gripper([0.25, 0.25])
        else:
            print("Move waist to down and gripper to 0.0")
            waist_state = [n - 5 for n in init_waist_state]
            print(f"Move Waist State: {waist_state}, {type(waist_state[0])}")
            robot.move_waist(waist_state)
            # robot.move_waist([n + 0.04 for n in init_waist_state])
            robot.move_gripper([0.0, 0.0])
        # robot.move_joints(BodySide.RIGHT, robot.arm_initial_joint_position[:7], speed=40, acc=3, block=True)
        round += 1

        time.sleep(5)

    robot.get_logger().info("Control task finished")

def parse_params(params):
    return [float(n) for n in params.split(",")]

def main(args=None):
    robot = Robot()
    stop_event = threading.Event()
    while not stop_event.is_set():
        try:
            user_input = input("Enter the part and params: ")
        except KeyboardInterrupt:
            print("Keyboard interrupt, shutting down node")
            stop_event.set()
            break
        splits = user_input.split(" ")
        params = ""
        if len(splits) > 1:
            part, params = splits
        else:
            part = splits[0]

        if part == "head" or part == "he":
            if params == "":
                print(f"Current head state: {robot.head_joint_states()}, usage: he <angle>,<angle>")
            else:
                params = parse_params(params)
                params = [n * (np.pi / 180) for n in params]
                print(f"Current head state: {robot.head_joint_states()}, move to {params}")
                robot.move_head(params)
        elif part == "waist" or part == "wa":
            if params == "":
                print(f"Current waist state: {robot.waist_joint_states()}, usage: wa <angle>,<height>")
            else:
                params = parse_params(params)
                params[0] = params[0] * (np.pi / 180)
                print(f"Current waist state: {robot.waist_joint_states()}, move to {params}")
                robot.move_waist(params)
        elif part == "hand" or part == "ha":
            if params == "":
                print(f"Current hand state: {robot.hand_joint_states()}")
            else:
                params = parse_params(params)
                print(f"Current hand state: {robot.hand_joint_states()}, move to {params}")
                robot.move_hand(params)
        elif part == "hand_as_gripper" or part == "ha_as_gr":
            if params == "":
                print(f"Current hand state: {robot.hand_joint_states()}")
            else:
                params = parse_params(params)
                print(f"Current hand state: {robot.hand_joint_states()}, move to {params}")
                robot.move_hand_as_gripper(params)
        elif part == "gripper" or part == "gr":
            if params == "":
                print(f"Current gripper state: {robot.gripper_states()}, usage: gr num,num")
            else:
                params = parse_params(params)
                print(f"Current gripper state: {robot.gripper_states()}, move to {params}")
                robot.move_gripper(params)
        elif part == "wheel" or part == "wh":
            if params == "":
                print(f"Move the wheel, usage: wh <speed> <angle> <round>")
            else:
                params = parse_params(params)
                print(f"Move wheel by {params}")
                for i in range(int(params[2])):
                    robot.move_wheel(params[0], params[1])
                    time.sleep(0.1)
        elif part == 'reset' or part == 're':
            print("Reset robot")
            robot.reset(arm_positions=robot.arm_initial_joint_position, head_positions=[0.0, 0.0], \
                        waist_positions=[0.0, 0.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
