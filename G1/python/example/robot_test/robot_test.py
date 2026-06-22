import time
import numpy as np
import ruckig
import cv2
from PIL import Image
import os
from a2d_sdk.robot import RobotDds
from a2d_sdk.robot import CosineCamera

# camera_names = [
#     "head",
#     "hand_left",
#     "hand_right",
#     "head_depth",
#     "hand_left_fisheye",
#     "hand_right_fisheye",
#     "back_left_fisheye",
#     "back_right_fisheye",
#     "head_center_fisheye",
#     "head_left_fisheye",
#     "head_right_fisheye"
# ]
class CameraTest:
    def __init__(self, camera_group, camera_names):
        self.camera_group = camera_group
        self.camera_names = camera_names
        self.save_root = "./test/camera_test"
        os.makedirs(self.save_root, exist_ok=True)

    def test_camera(self):
        for i in range(10):
            for camera_name in self.camera_names:
                camera_image, camera_img_ts = self.camera_group.get_latest_image(camera_name)
                width, height = self.camera_group.get_image_shape(camera_name)
                if camera_image is not None:
                    if "depth" in camera_name:
                        camera_image = np.frombuffer(camera_image, dtype=np.uint16)
                        camera_image = camera_image.reshape(width, height, 1)
                        camera_image = cv2.resize(camera_image, (width,height))
                        cv2.imwrite(f"{self.save_root}/{camera_name}_{camera_img_ts}_{i}.png", camera_image)
                    else:
                        camera_image = Image.fromarray(camera_image)
                        camera_image.save(f"{self.save_root}/{camera_name}_{camera_img_ts}_{i}.jpeg")
                else:
                    print("camera_img is none")
                    continue

class ArmTest:
    def __init__(self,robot):
        """Initialize test environment"""
        self.robot = robot
        self.arm_initial_joint_position = [
            0.881, 0.671, -0.5369, -1.8702, 0.8555, 0.6907, 1.1398,  # left arm
            -0.881, -0.671, 0.5369, 1.8702, -0.8555, -0.6907, -1.1398  # right arm
        ]
        time.sleep(1)
    def test_arm_trajectory_planning(self, target_positions=None):
        """Test arm trajectory planning and execution

        Args:
            target_positions: Optional target joint positions. If None, uses initial positions.
        """
        print("Starting arm trajectory test...")

        # Get current joint states
        current_positions, _ = self.robot.arm_joint_states()
        if not current_positions:
            raise Exception("Failed to get arm joint states")
        if target_positions is None:
            target_positions = self.arm_initial_joint_position[:14]

        print(f"Planning trajectory from current positions to target...")
        print(f"Current positions: {current_positions}")
        print(f"Target positions: {target_positions}")
        dof = 14  # Degrees of freedom
        interval = 0.01  # Time interval
        rk = ruckig.Ruckig(dof, interval)
        rk_input = ruckig.InputParameter(dof)
        rk_output = ruckig.OutputParameter(dof)

        # Set current state
        rk_input.current_position = current_positions
        rk_input.current_velocity = [0.0] * dof
        rk_input.current_acceleration = [0.0] * dof

        # Set target state
        rk_input.target_position = target_positions
        rk_input.target_velocity = [0.0] * dof
        rk_input.target_acceleration = [0.0] * dof

        # Set motion constraints
        rk_input.max_velocity = [2.0] * dof
        rk_input.max_acceleration = [1.0] * dof
        rk_input.max_jerk = [5.0] * dof

        # Generate trajectory
        print("Generating trajectory...")
        trajs = []
        while rk.update(rk_input, rk_output) == ruckig.Result.Working:
            trajs.append(rk_output.new_position)
            rk_output.pass_to_input(rk_input)
        print(f"Generated {len(trajs)} trajectory points")
        print("Executing trajectory...")
        for i, traj in enumerate(trajs):
            print(f"Executing point {i+1}/{len(trajs)}")
            self.robot.move_arm(traj)
            time.sleep(interval)
        time.sleep(0.5)

        print("Trajectory test completed successfully!")

class HeadTest:
    def __init__(self,robot):
        """初始化头部测试环境"""
        self.robot = robot
        self.head_initial_joint_positions = [0.0, 0.0]
        time.sleep(1)

    def test_head_motion(self, target_positions=None):
        """
        测试头部运动，直接将头部移动到目标位置。

        参数:
            target_positions: 目标关节位置列表。如果为 None，则使用初始位置。
        """
        print("开始头部运动测试...")
        # 获取当前头部关节状态
        current_positions, _ = self.robot.head_joint_states()
        if not current_positions:
            raise Exception("获取头部关节状态失败")
        if target_positions is None:
            target_positions = self.head_initial_joint_positions
        print(f"当前头部位置: {current_positions}")
        print(f"目标头部位置: {target_positions}")
        self.robot.move_head(target_positions)
        time.sleep(2)
        print("头部运动完成。\n")


class WaistTest:
    def __init__(self,robot):
        """初始化腰部测试环境"""
        self.robot = robot
        # 假设腰部有2个自由度，可以根据实际情况调整初始角度
        self.waist_initial_positions = [0.5, 20]
        time.sleep(1)

    def test_waist_motion(self, target_positions=None):
        """
        测试腰部运动，直接将腰部移动到目标位置。

        Args:
            target_positions: 目标关节位置列表。如果为 None，则使用初始位置。
        """
        print("开始腰部运动测试...")
        current_positions, _ = self.robot.waist_joint_states()
        if not current_positions:
            raise Exception("获取腰部关节状态失败")
        if target_positions is None:
            target_positions = self.waist_initial_positions
        print(f"当前腰部位置: {current_positions}")
        print(f"目标腰部位置: {target_positions}")
        self.robot.move_waist(target_positions)
        time.sleep(2)
        print("腰部运动完成。\n")
        
class GripperTest:
    def __init__(self,robot):
        """Initialize test environment for gripper"""
        self.robot = robot
        self.gripper_initial_positions = [0.0, 0.0]
        time.sleep(1)

    def test_gripper_motion(self, target_positions=None):
        """
        Test gripper motion by directly moving to the target position.

        Args:
            target_positions: Target positions for gripper. If None, uses initial positions.
        """
        print("开始夹爪运动测试...")
        current_positions, _ = self.robot.gripper_states()
        if not current_positions:
            raise Exception("获取夹爪状态失败")
        if target_positions is None:
            target_positions = self.gripper_initial_positions
        print(f"当前夹爪位置: {current_positions}")
        print(f"目标夹爪位置: {target_positions}")
        self.robot.move_gripper(target_positions)
        time.sleep(2)
        print("夹爪运动完成。\n")

class WheelTest:
    def __init__(self,robot):
        """初始化底盘测试环境"""
        self.robot = robot
        # 初始状态，这里不再作为目标状态而是作为复位默认值
        self.wheel_initial_positions = [0.0, 0.0, 0]  # 例如速度、角度和轮次，轮次可设为0表示不运动
        time.sleep(1)

    def test_wheel_motion(self, target_positions=None):
        """
        测试底盘运动，按照如下逻辑控制：
            - target_positions 为一个包含三个元素的列表 [speed, angle, round]
            - 依次执行 round 次，每次调用 robot.move_wheel(speed, angle) 后间隔 0.1 秒
        如果不传入目标，则使用初始状态（即不运动）。
        """
        print("开始底盘运动测试...")
        
        if target_positions is None:
            target_positions = self.wheel_initial_positions
        
        print(f"目标底盘参数: {target_positions}")
        
        # 按照 target_positions 中的参数进行控制：
        # target_positions: [speed, angle, round]
        rounds = int(target_positions[2])
        for i in range(rounds):
            self.robot.move_wheel(target_positions[0], target_positions[1])
            time.sleep(0.1)
        print("底盘运动完成。\n")


def main():
    robot = None
    camera_group = None
    try:
        robot = RobotDds()
        camera_names = ["head", "hand_left", "hand_right", "head_depth", "hand_left_depth", "hand_right_depth"]
        camera_group = CosineCamera(camera_names)
        # 先打印提示信息，再等待用户输入 "c" 进行下一步操作
        print("注意，手臂测试活动范围较大，请确保本体2米范围内无障碍物...")
        user_input = input("请输入 'c' 继续后续操作: ")
        while user_input.lower() != 'c':
            user_input = input("请输入 'c' 来继续: ")
        camera_test = CameraTest(camera_group, camera_names)
        arm_test = ArmTest(robot)
        head_test = HeadTest(robot)
        waist_test = WaistTest(robot)
        gripper_test = GripperTest(robot)
        wheel_test = WheelTest(robot)
        
        # 循环读取用户输入，根据用户输入选择所有测试
        while True:
            print("请输入 'all' 来测试所有运动、'camera' 来测试相机、'q' 退出")
            print("请输入 'arm' 来测试手臂运动、'head' 来测试头部运动、'waist' 来测试腰部运动、'gripper' 来测试夹爪运动、'wheel' 来测试车轮运动")
            user_input = input("请输入: ")
            if user_input.lower() == 'all':
                camera_test.test_camera()

                arm_test.test_arm_trajectory_planning()
                custom_arm_positions = [0.0] * 14
                arm_test.test_arm_trajectory_planning(custom_arm_positions)

                head_test.test_head_motion()
                custom_head_positions = [0.2, 0.2]
                head_test.test_head_motion(custom_head_positions)
                head_test.test_head_motion()

                waist_test.test_waist_motion()
                custom_waist_positions = [0.2, 35]
                waist_test.test_waist_motion(custom_waist_positions)
                waist_test.test_waist_motion()

                gripper_test.test_gripper_motion() 
                custom_gripper_positions = [1, 1]  
                gripper_test.test_gripper_motion(custom_gripper_positions)
                gripper_test.test_gripper_motion()  

                wheel_test.test_wheel_motion()
                custom_wheel_positions = [0.1, 0.0, 20]  # 示例目标，可根据需求调整
                wheel_test.test_wheel_motion(custom_wheel_positions)

                print("所有测试执行完毕")
            elif user_input.lower() == 'camera':
                camera_test.test_camera()
                print("相机测试完成")
            elif user_input.lower() == 'arm':
                arm_test.test_arm_trajectory_planning()
                custom_arm_positions = [0.0] * 14
                arm_test.test_arm_trajectory_planning(custom_arm_positions)
                print("手臂测试完成")
            elif user_input.lower() == 'head':
                head_test.test_head_motion()
                custom_head_positions = [0.2, 0.2]
                head_test.test_head_motion(custom_head_positions)
                head_test.test_head_motion()
                print("头部测试完成")
            elif user_input.lower() == 'waist':
                waist_test.test_waist_motion()
                custom_waist_positions = [0.2, 35]
                waist_test.test_waist_motion(custom_waist_positions)
                waist_test.test_waist_motion()
                print("腰部测试完成")
            elif user_input.lower() == 'gripper':
                gripper_test.test_gripper_motion()
                custom_gripper_positions = [1, 1]  
                gripper_test.test_gripper_motion(custom_gripper_positions)
                gripper_test.test_gripper_motion()
                print("夹爪测试完成")
            elif user_input.lower() == 'wheel':
                wheel_test.test_wheel_motion()
                custom_wheel_positions = [0.1, 0.0, 20]  # 示例目标，可根据需求调整
                wheel_test.test_wheel_motion(custom_wheel_positions)
                print("车轮测试完成")
            elif user_input.lower() == 'q':
                break

    except Exception as e:
        print(f"Test failed: {str(e)}")
        return 1
    finally:
        # 释放 robot 对象资源
        if robot:
            robot.shutdown()
    return 0

if __name__ == "__main__":
    exit(main())