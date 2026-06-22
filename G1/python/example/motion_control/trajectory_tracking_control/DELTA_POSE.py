#!/usr/bin/env python3
import time
from a2d_sdk.robot import RobotController, RobotDds

def poll_state(getter, length, name, timeout=2.0, interval=0.1):
    deadline = time.time() + timeout
    last_vals = None

    while time.time() < deadline:
        vals, _ = getter()
        last_vals = vals
        # 获取长度
        try:
            actual_len = len(vals)
        except Exception:
            actual_len = None

        # 判断每个元素是否可转 float
        all_numeric = (actual_len == length) and all(
            _is_number(v) for v in vals
        )
        if all_numeric:
            print(f"✅ {name} 就绪")
            return list(vals)

        time.sleep(interval)

    raise RuntimeError(f"{name} 在 {timeout}s 内未就绪，最后 vals={last_vals!r}")

def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def execute(rc: RobotController, action: dict):
    """
    调用 SDK 下发相对位姿轨迹控制命令。
    """
    # 定义机器人状态字典
    robot_states = {
        "head":  action["head_joint_states"],
        "waist": action["waist_joint_states"],
        "arm":   action["arm_joint_states"],
    }
    # 定义机器人动作列表
    robot_actions = [
        {
            "left_arm":  {"action_data": delta[:6],  "control_type": "DELTA_POSE"},
            "right_arm": {"action_data": delta[6:12], "control_type": "DELTA_POSE"},
        }
        for delta in action["arm_cmd"]
    ]
    # 调用 SDK 下发相对位姿轨迹控制命令
    rc.trajectory_tracking_control(
        infer_timestamp      = action["observation_timestamp"],#必须有的时间
        robot_states         = robot_states,# 必须的参考状态，有了更好
        robot_actions        = robot_actions,
        robot_link           = "base_link",#基础坐标系
        trajectory_reference_time = 1.0,#运行的参考时间，越小运行的越快
    )

def main():
    rc = RobotController()
    rd = RobotDds()
    time.sleep(2.0) 
    head_states  = poll_state(rd.head_joint_states,  length=2,  name="head_joint_states")
    waist_states = poll_state(rd.waist_joint_states, length=2,  name="waist_joint_states")
    arm_states   = poll_state(rd.arm_joint_states,   length=14, name="arm_joint_states")

    print(">>> 所有传感器数据就绪，开始下发相对位姿控制")
    try:
        while True:
            ts_ns = int(time.time() * 1e9)
            action = {
                "observation_timestamp": ts_ns,
                "head_joint_states":     head_states,
                "waist_joint_states":    waist_states,
                "arm_joint_states":      arm_states,
                "arm_cmd": [
                    [0.02, 0, 0, 0, 0, 0,   0.02, 0, 0, 0, 0, 0]
                ],
            }
            execute(rc, action)
            print(f">>> [{time.strftime('%H:%M:%S')}] 相对位姿控制命令已下发")
            time.sleep(1.0)  # 到下一次推理开始的等待时间
    except KeyboardInterrupt:
        print("\n>>> 收到中断，退出。")
    finally:
        rd.shutdown()

if __name__ == "__main__":
    main()