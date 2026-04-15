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
        os.makedirs(log_dir, exist_ok=True)
        
        if filepath is None or filepath == "leader_arm_qc_log.txt":
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filepath = os.path.join(log_dir, f"{now_str}_{filepath if filepath else 'log'}.txt")
        else:
            self.filepath = os.path.join(log_dir, filepath)

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")




def main(address, model):
    logger = File_Logger()
    robot = rby.create_robot(address, model)
    robot.connect()

    if not robot.is_connected():
        print("Error: Robot connection failed.")
        exit(1)

    if not robot.power_on("12v"):
        print("Error: Failed to power on 12V.")
        exit(1)

    leader_arm = LeaderArm(control_period=0.01)

    def handler(signum, frame):
        print("\nInterrupt received. Stopping...")
        if leader_arm:
            leader_arm.close()
        robot.power_off("12v")
        exit(1)
    
    signal.signal(signal.SIGINT, handler)
    leader_arm.initialize(verbose=True)
    
    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print("Error: Mismatch in the number of devices detected for RBY Master Arm.")
        exit(1)

    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    def control(state: LeaderArm.State):
        header = f"--- Leader Arm state Monitor | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q = f"q (rad):      {fmt(state.q_joint)}"
        line_temp = f"temp (C):     {fmt(state.temperatures)}"
        line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn = f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d} | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"

        # 5. Status & Alarm Section (Fixed position at bottom)
        if state.fault_ids or state.tool_fault_ids:
            all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
            status_line = f"\033[1;31mSTATUS: [ !! CRITICAL ALARM !! - FAILED IDs: {all_faults} ]\033[0m"
        elif state.tool_warning_ids:
            # Transient warning (1-4 consecutive failures)
            status_line = f"\033[1;33mSTATUS: [ WARNING - Comm jitter on IDs: {state.tool_warning_ids} ]\033[0m"
        else:
            status_line = "\033[1;32mSTATUS: [ NORMAL ]\033[0m"

        print("\033[H\033[J", end="")  # Clear terminal and move cursor to top
        print(header)
        print(line_idx)
        print(line_q)
        print(line_temp)
        print(line_torque)
        print(line_grav)
        print(line_btn)
        print("\n" + status_line)

        # Tool Warning
        if state.tool_fault_ids:
            warning_msg = f"! [TOOL WARNING] Communication failure on IDs: {state.tool_fault_ids}"
            print(warning_msg)
            logger.save(f"{warning_msg}\n")

        # Log to file
        logger.save(f"{header}\n{line_q}\n{line_temp}\n{line_torque}\n{line_grav}\n{line_btn}\n, {status_line}\n")

        input = LeaderArm.ControlInput()

        input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        input.target_torque = state.gravity_term

        return input

    def safety_function(state: LeaderArm.State):
        all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR] Communication failure detected on IDs: {all_faults}\033[0m\nACTION: Immediate Emergency Shutdown.\n"
        print(error_msg)
        logger.save(error_msg)
        
        if leader_arm:
            leader_arm.close()
        robot.power_off("12v")
        os._exit(1)

    leader_arm.start_control(control, safety_function=safety_function)

    time.sleep(100)

    leader_arm.stop_control()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="19_master_arm")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model)
