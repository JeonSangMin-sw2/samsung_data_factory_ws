import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal
import sys

from leader_arm import LeaderArm



# 해당코드는 미완성. 모터 포지션 입력값이 제대로 들어가나 확인할려고 만든 코드인데 제대로 작동 안함
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
        targets[2][1][mid] += OFFSET_RAD

        for label, q_target in targets:
            # Manual S-curve interpolation loop
            start_time = time.time()
            current_q_start = leader_arm.state.q_joint.copy()
            
            # Ensure mode is current-based position
            leader_arm.set_target_position(current_q_start, goal_current=0.5)
            
            while True:
                elapsed = time.time() - start_time
                if elapsed >= MOVE_DURATION:
                    leader_arm.set_target_position(q_target, goal_current=0.5)
                    break
                
                # S-curve interpolation
                t = elapsed / MOVE_DURATION
                alpha = (1.0 - np.cos(np.pi * t)) / 2.0
                interp_q = current_q_start + (q_target - current_q_start) * alpha
                
                # Direct bus write (via leader_arm helper if we had one, but we use it via bus)
                # We use set_target_position for mode but for real-time we can use raw write
                leader_arm.bus.group_sync_write_send_torque([(i, 0.5) for i in range(leader_arm.DOF)])
                leader_arm.bus.group_sync_write_send_position([(i, float(q)) for i, q in enumerate(interp_q)])
                
                # Monitoring
                curr_state = leader_arm.state.copy()
                q_val = curr_state.q_joint[mid]
                sys.stdout.write(f"\r  --> Moving to {label}: Pos: {q_val:7.3f} rad | Current: {curr_state.current[mid]:7.3f} A  \033[K")
                sys.stdout.flush()
                time.sleep(0.01)
            
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
