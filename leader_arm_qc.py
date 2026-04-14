import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal

from leader_arm import LeaderArm

URDF_PATH = "/models/master_arm/model.urdf"

# 데이터를 텍스트 파일로 저장하는 클래스. 디버깅을 위함
class File_Logger:
    def __init__(self, filepath=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "log")
        
        if filepath is None or filepath == "leader_arm_qc_log.txt":
            now_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filepath = os.path.join(log_dir, f"{now_str}.txt")
        else:
            self.filepath = os.path.join(log_dir, filepath)

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")




def main(address, model):
    robot = rby.create_robot(address, model)
    robot.connect()

    if not robot.is_connected():
        print("Error: Robot connection failed.")
        exit(1)

    if not robot.power_on("12v"):
        print("Error: Failed to power on 12V.")
        exit(1)

    def handler(signum, frame):
        robot.power_off("12v")
        exit(1)
    
    signal.signal(signal.SIGINT, handler)

    leader_arm = LeaderArm()
    leader_arm.initialize()
    
    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print("Error: Mismatch in the number of devices detected for RBY Master Arm.")
        exit(1)

    def control(state: LeaderArm.State):
        temperature = []
        with np.printoptions(suppress=True, precision=3, linewidth=300):
            print(f"--- {datetime.datetime.now().time()} ---")
            print(f"q: {state.q_joint}")
            print(f"temperature: {state.temperatures}")
            print(f"g: {state.gravity_term}")
            print(
                f"right: {state.button_right.button}, left: {state.button_left.button}"
            )

        input = LeaderArm.ControlInput()

        input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        input.target_torque = state.gravity_term

        return input

    leader_arm.start_control(control)

    time.sleep(100)

    master_arm.stop_control()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="19_master_arm")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model)
