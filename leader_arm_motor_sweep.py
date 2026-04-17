import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal
import sys

from leader_arm import LeaderArm

def main(address, model):
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
        print("\n\n[System] Interrupt received. Shutting down...")
        if leader_arm:
            leader_arm.close()
        robot.power_off("12v")
        os._exit(0)
    
    signal.signal(signal.SIGINT, handler)
    
    if not leader_arm.initialize(verbose=False):
        print("Error: Leader Arm initialization failed.")
        exit(1)

    # Display settings
    MOVE_DURATION = 1.0
    OFFSET_DEG = 5.0
    OFFSET_RAD = np.deg2rad(OFFSET_DEG)

    def control(state: LeaderArm.State):
        # We just keep gravity compensation active during the whole process
        input_data = LeaderArm.ControlInput()
        input_data.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        input_data.target_torque = state.gravity_term
        return input_data

    leader_arm.start_control(control)

    print(f"\n{'='*60}")
    print(f"  Leader Arm Motor Sweep Diagnostic Tool")
    print(f"{'='*60}\n")

    active_joints = leader_arm.active_joint_ids
    print(f"[Info] Detected Active Joints: {active_joints}\n")

    for mid in active_joints:
        print(f"[Joint {mid:2d}] Starting sweep...")
        
        # Get current state
        state = leader_arm.state.copy()
        q_base = state.q_joint.copy()
        
        targets = [
            ("-5 deg", q_base.copy()),
            (" 0 deg", q_base.copy()),
            ("+5 deg", q_base.copy())
        ]
        targets[0][1][mid] -= OFFSET_RAD
        # targets[1] is base, no change
        targets[2][1][mid] += OFFSET_RAD

        for label, q_target in targets:
            # Inline monitoring while moving
            start_time = time.time()
            leader_arm.set_target_position(q_target, duration=MOVE_DURATION)
            
            while time.time() - start_time < MOVE_DURATION + 0.5:
                curr_state = leader_arm.state.copy()
                q_val = curr_state.q_joint[mid]
                c_val = curr_state.current[mid]
                t_val = curr_state.temperatures[mid]
                
                # Real-time state line
                sys.stdout.write(f"\r  --> Moving to {label}: Pos: {q_val:7.3f} rad | Current: {c_val:7.3f} A | Temp: {t_val:5.1f} C  \033[K")
                sys.stdout.flush()
                time.sleep(0.05)
            
            print() # New line after finishing one target

        print(f"[Joint {mid:2d}] Sweep OK.\n")

    print(f"{'='*60}")
    print("  All diagnostic sweeps completed successfully.")
    print(f"{'='*60}\n")

    leader_arm.close()
    robot.power_off("12v")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leader Arm Motor Sweep Diagnostic")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="a", help="Robot Model Name")
    args = parser.parse_args()
    main(args.address, args.model)
