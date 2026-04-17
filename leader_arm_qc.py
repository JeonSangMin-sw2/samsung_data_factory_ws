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
DEFAULT_WAIT_TIME = 2.0
POSTURE_ERROR_THRESHOLD = 0.15 # ~8.6 degrees

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

def verify_saved_positions():
    positions = load_positions()
    if positions is not None:
        print(f"\n[Verification] Reading from {POSITION_FILE}:")
        for i, pos in enumerate(positions):
            print(f"  #{i+1:2d}: {', '.join([f'{x:7.3f}' for x in pos])}")
        print(f"[Verification] Total {len(positions)} positions verified.\n")

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
    fault_registry = {} # Tracks {id: cumulative_error_count}

    def handler(signum, frame):
        nonlocal recorded_positions
        print("\n\n[System] Interrupt received. Finalizing session...")
        if mode == 'capture':
            if recorded_positions:
                save_positions(recorded_positions)
                verify_saved_positions()
            else:
                print("[Info] No positions were recorded in this session.")
        
        if leader_arm:
            print("[System] Closing Leader Arm engine...")
            leader_arm.close()
        
        print("[System] Powering off 12V and exiting.")
        robot.power_off("12v")
        time.sleep(0.5)
        os._exit(0)
    
    signal.signal(signal.SIGINT, handler)
    leader_arm.initialize(verbose=True)
    
    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print(f"Error: Mismatch in the number of devices detected. Expected {leader_arm.DEVICE_COUNT}, got {len(leader_arm.active_ids)}")
        exit(1)

    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    def control(state: LeaderArm.State):
        nonlocal last_btn_state, fault_registry
        
        # In check/capture mode, skip printing if we detected a fault
        if (mode == 'check' or mode == 'capture') and not check_status['is_ok']:
            return LeaderArm.ControlInput()

        header = f"--- Leader Arm QC Monitor [{mode.upper()}] | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q = f"q (rad):      {fmt(state.q_joint)}"
        line_current = f"current (A):  {fmt(state.current)}"
        line_temp = f"temp (C):     {fmt(state.temperatures)}"
        line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn = f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d} | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"

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
        print(line_current)
        print(line_temp)
        print(line_torque)
        print(line_grav)
        print(line_btn)
        if mode == 'capture':
            print(f"Captured:     {len(recorded_positions)}")

        # Log to file
        logger.save(f"{header}\n{line_idx}\n{line_q}\n{line_current}\n{line_temp}\n{line_torque}\n{line_grav}\n{line_btn}\n")

        input_data = LeaderArm.ControlInput()
        
        # 6. Control Logic by Mode
        if mode == 'capture':
            # Use CurrentControlMode with gravity compensation for easy manual movement
            input_data.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            input_data.target_torque = state.gravity_term
        else:
            # Check mode: Use CurrentBasedPositionControlMode for precise playback
            # We use MAXIMUM_TORQUE as a limit to ensure it can overcome friction.
            input_data.target_operating_mode.fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            input_data.target_torque.fill(leader_arm.MAXIMUM_TORQUE)
            
        return input_data

    def safety_function(state: LeaderArm.State):
        nonlocal check_status, fault_registry
        
        check_status['is_ok'] = False
        for mid in state.fault_ids:
            fault_registry[mid] = fault_registry.get(mid, 0) + 1
        
        failed_list = sorted(list(state.fault_ids))
        registry_summary = ", ".join([f"ID {mid}(x{fault_registry[mid]})" for mid in sorted(fault_registry.keys())])
        
        error_msg = f"\n\n[SAFETY TRIGGERED] Real-time communication failure!\nFailed IDs: {failed_list}\nCumulative registry: {registry_summary}\n"
        print(error_msg)
        logger.save(error_msg)
        
        print("\nACTION: Executing Soft Shutdown (Torque Fade-out for 2s)...")
        
        # Capture the last calculated gravity compensation
        initial_torque = state.gravity_term.copy()
        steps = 100
        interval = 2.0 / steps
        
        for i in range(steps):
            scale = 1.0 - (i / steps)
            # Apply scaled torque to all motors that are still responsive
            id_torque = []
            for mid in range(leader_arm.DOF):
                if mid not in state.fault_ids:
                    id_torque.append((mid, initial_torque[mid] * scale))
            
            if id_torque:
                leader_arm.bus.group_sync_write_send_torque(id_torque)
            
            time.sleep(interval)

        print("Soft Shutdown complete. Powering off 12V.")
        if leader_arm:
            leader_arm.close()
        robot.power_off("12v")
        os._exit(1) # Immediate process termination

    if mode == 'capture':
        leader_arm.start_control(control, safety_function=safety_function)
        print("Capture mode active. Press any button to record posture. Ctrl+C to save and exit.")
        while True:
            time.sleep(1)
    else: # check mode
        positions = load_positions()
        if positions is not None:
            # Start control loop with both control and safety callbacks
            leader_arm.start_control(control, safety_function=safety_function)
            
            print(f"\n--- Starting Status Check Sequence ({len(positions)} postures) ---")
            active_ids = leader_arm.active_ids
            
            for i, pos in enumerate(positions):
                check_status['pos_idx'] = i
                check_status['is_ok'] = True # Reset OK status for the new posture
                
                print(f"\n[Check {i+1}/{len(positions)}] Moving to posture...")
                leader_arm.set_target_position(pos,duration=1.0)
                
                # Wait for the posture stabilization period
                time.sleep(DEFAULT_WAIT_TIME)
                
                # Check tracking error (verify if the arm actually reached the target)
                state = leader_arm.state.copy()
                error = np.abs(state.q_joint - pos)
                failed_indices = np.where(error > POSTURE_ERROR_THRESHOLD)[0]
                
                if len(failed_indices) > 0:
                    error_msg = f"\n\n[POSITIONING FAILURE] Motor(s) too weak to reach posture {i+1}!\n"
                    for idx in failed_indices:
                        error_msg += f"  - Joint ID {idx}: Error {error[idx]:.4f} rad (Target: {pos[idx]:.4f}, Current: {state.q_joint[idx]:.4f})\n"
                    print(error_msg)
                    logger.save(error_msg)
                    
                    # Trigger soft shutdown for safety
                    safety_function(state)
                    return

                print(".", end="", flush=True)
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
