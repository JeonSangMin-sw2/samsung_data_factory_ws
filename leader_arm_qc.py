import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal

from leader_arm import LeaderArm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSITION_FILE = os.path.join(DATA_DIR, "position_list.npz")
DEFAULT_WAIT_TIME = 5.0

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

def save_positions(positions):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    np.savez(POSITION_FILE, positions=np.array(positions))
    print(f"\n[Info] Saved {len(positions)} positions to {POSITION_FILE}")

def load_positions():
    if not os.path.exists(POSITION_FILE):
        print(f"[Error] Position file not found: {POSITION_FILE}")
        return None
    data = np.load(POSITION_FILE)
    return data['positions']

def main(address, model, mode):
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
    
    recorded_positions = []
    last_btn_state = {'right': 0, 'left': 0}
    
    # Shared state for monitoring and debugging
    check_status = {'is_ok': True, 'pos_idx': -1}

    def handler(signum, frame):
        print("\nInterrupt received. Stopping...")
        if mode == 'capture' and recorded_positions:
            save_positions(recorded_positions)
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
        nonlocal last_btn_state
        
        # In check mode, skip printing if we detected a fault (to keep the error message visible)
        if mode == 'check' and not check_status['is_ok']:
            return LeaderArm.ControlInput()

        header = f"--- Leader Arm QC Monitor [{mode.upper()}] | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q = f"q (rad):      {fmt(state.q_joint)}"
        line_temp = f"temp (C):     {fmt(state.temperatures)}"
        line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn = f"buttons:      right: {state.button_right.button:1d}, left: {state.button_left.button:1d}"

        # Capture logic
        if mode == 'capture':
            curr_btn_right = state.button_right.button
            curr_btn_left = state.button_left.button
            
            # Record on rising edge
            if (curr_btn_right == 1 and last_btn_state['right'] == 0) or \
               (curr_btn_left == 1 and last_btn_state['left'] == 0):
                recorded_positions.append(state.q_joint.copy())
                print(f"\n[Capture] Recorded position #{len(recorded_positions)}")
            
            last_btn_state['right'] = curr_btn_right
            last_btn_state['left'] = curr_btn_left

        # Conditional Clearing and Printing
        # print("\033[H\033[J", end="")  # Clear terminal and move cursor to top
        print(header)
        
        if mode == 'check' and check_status['pos_idx'] >= 0:
            print(f"Status:       position {check_status['pos_idx'] + 1} is ok")
        
        print(line_idx)
        print(line_q)
        print(line_temp)
        print(line_torque)
        print(line_grav)
        print(line_btn)
        if mode == 'capture':
            print(f"Captured:     {len(recorded_positions)}")

        # Log to file
        logger.save(f"{header}\n{line_idx}\n{line_q}\n{line_temp}\n{line_torque}\n{line_grav}\n{line_btn}\n")

        input_data = LeaderArm.ControlInput()
        
        # Capture mode uses CurrentControlMode (gravity compensation)
        if mode == 'capture':
            input_data.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        # Check mode: We do NOT force CurrentControlMode here to allow set_target_position 
        # to correctly switch to CurrentBasedPositionControlMode without jitter.
        
        input_data.target_torque = state.gravity_term

        return input_data

    if mode == 'capture':
        leader_arm.start_control(control)
        print("Capture mode active. Press any button to record posture. Ctrl+C to save and exit.")
        while True:
            time.sleep(1)
    else: # check mode
        positions = load_positions()
        if positions is not None:
            # Start control loop for monitoring and gravity compensation (using current position if idle)
            leader_arm.start_control(control)
            
            print(f"\n--- Starting Status Check Sequence ({len(positions)} postures) ---")
            active_ids = leader_arm.active_ids
            
            for i, pos in enumerate(positions):
                check_status['pos_idx'] = i
                check_status['is_ok'] = True # Reset OK status for the new posture
                
                print(f"\n[Check {i+1}/{len(positions)}] Moving to posture...")
                leader_arm.set_target_position(pos)
                
                # Wait and monitor health
                is_ok, failed_id = leader_arm.monitor_health(DEFAULT_WAIT_TIME)
                if not is_ok:
                    check_status['is_ok'] = False
                    # Screen will stop updating via 'control' callback, allowing this message to persist
                    print(f"\n\n[CRITICAL ERROR] ID {failed_id} connection check failed at posture {i+1}!")
                    input("Press Enter to acknowledge and exit...")
                    leader_arm.close()
                    robot.power_off("12v")
                    exit(1)
            
            leader_arm.stop_control()
            print("\n\n--- Status Check Sequence Completed Successfully ---")
        else:
            print("Check mode failed: No positions to check.")

    leader_arm.close()
    robot.power_off("12v")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leader Arm QC Monitor")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--mode", type=str, default="check", choices=["check", "capture"], help="Operation mode: check (default) or capture"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model, mode=args.mode)
