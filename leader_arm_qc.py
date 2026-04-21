"""
Leader Arm QC Test

리더암이 스스로 N 번의 사이클 동안 저장된 position을 순회하며 움직입니다.
또한 'capture' 모드를 통해 사용자가 리더암을 움직여 position을 기록할 수 있습니다.

주요 기능:
  - 실시간 상태 모니터링 (teleop_with_leader_arm.py 참고)
  - --mode capture: 버튼을 눌러 현재 자세를 기록하고 npz 파일로 저장
  - --mode check: 기록된 position을 Position 모드로 순회하며 품질 검사
  - 로봇 쪽으로 제어 명령을 보내지 않음 (리더암만 움직임)
  - 통신 장애 발생 시 12V 차단 safety 적용

References:
  - teleop_with_leader_arm.py : 실시간 모니터링 + position 모드 제어
  - leader_arm_state_check.py : safety_function 패턴
"""

import os
import time
import signal
import argparse
import datetime
import numpy as np
import rby1_sdk as rby

from leader_arm import LeaderArm


# ============================================================
# Configuration
# ============================================================
DEFAULT_CYCLES = 5
SETTLE_THRESHOLD = 0.10       # rad (~5.7 deg) — position 도달 판정 임계값
SETTLE_DURATION = 0.5         # sec — 임계값 이내로 유지해야 하는 시간
POSITION_TIMEOUT = 3        # sec — 한 position에서 대기하는 최대 시간
TORQUE_LIMIT = np.array([1.5, 1.5, 1.5, 1.5, 0.6, 0.6, 0.6] * 2)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSITION_FILE = os.path.join(DATA_DIR, "position_list.npz")
NPZ_VALID_DOF = LeaderArm.DOF


# ============================================================
# File Helpers
# ============================================================
class FileLogger:
    def __init__(self, filepath=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        if filepath is None:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filepath = os.path.join(log_dir, f"{now_str}_qc_test.txt")
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
        raise FileNotFoundError(f"Position file not found: {POSITION_FILE}")

    with np.load(POSITION_FILE) as data:
        raw_positions = np.asarray(data['positions'], dtype=np.float64)

    if raw_positions.ndim == 1:
        raw_positions = raw_positions.reshape(1, -1)

    if raw_positions.shape[1] < NPZ_VALID_DOF:
        raise ValueError(
            f"Invalid position data shape {raw_positions.shape}: "
            f"expected at least {NPZ_VALID_DOF} columns"
        )
    if raw_positions.shape[0] == 0:
        raise ValueError("Position file does not contain any positions")

    return raw_positions[:, :NPZ_VALID_DOF].copy()

def verify_saved_positions():
    positions = load_positions()
    if positions is not None:
        print(f"\n[Verification] Reading from {POSITION_FILE}:")
        for i, pos in enumerate(positions):
            print(f"  #{i+1:2d}: {', '.join([f'{x:7.3f}' for x in pos])}")
        print(f"[Verification] Total {len(positions)} positions verified.\n")


# ============================================================
# Main
# ============================================================
def main(address, model, num_cycles, mode):
    logger = FileLogger()

    positions = load_positions() if mode == 'check' else None
    num_positions = len(positions) if positions is not None else 0

    # ===== SETUP ROBOT (12V 공급만 사용) =====
    robot = rby.create_robot(address, model)
    robot.connect()

    if not robot.is_connected():
        print("Error: Robot connection failed.")
        exit(1)

    if not robot.power_on("12v"):
        print("Error: Failed to power on 12V.")
        exit(1)

    # ===== LEADER ARM SETUP =====
    leader_arm = LeaderArm(control_period=0.01)
    leader_arm.set_max_retries(max_tool_retries=100, max_joint_retries=100)

    if not leader_arm.initialize(verbose=True):
        print("Failed to initialize Leader Arm")
        exit(1)

    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print(
            f"Error: Mismatch in the number of devices detected. "
            f"Expected {leader_arm.DEVICE_COUNT}, got {len(leader_arm.active_ids)}"
        )
        exit(1)

    # ===== QC TEST STATE =====
    recorded_positions = []
    last_btn_state = {'right': 0, 'left': 0}

    qc_state = {
        "current_pos_idx": 0,
        "cycle_count": 0,
        "total_cycles": num_cycles,
        "settle_start": None,
        "pos_start_time": time.time(),
        "pos_reached": False,
        "test_complete": False,
        "pos_timeout_count": 0,
        "total_pos_visited": 0,
    }

    session_stats = {
        "total_warnings": 0,
        "max_streak": 0,
    }

    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    # =========================================================
    # CONTROL CALLBACK
    # =========================================================
    def control(state: LeaderArm.State):
        # --------------------------------------------------
        # 1. Capture 모드 로직
        # --------------------------------------------------
        if mode == 'capture':
            curr_btn_right = state.button_right.button
            curr_btn_left = state.button_left.button
            
            # 버튼 누름(Rising Edge) 감지 시 기록
            if (curr_btn_right == 1 and last_btn_state['right'] == 0) or \
               (curr_btn_left == 1 and last_btn_state['left'] == 0):
                recorded_positions.append(state.q_joint.copy())
                print(f"\n[Capture] Recorded position #{len(recorded_positions)}")
            
            last_btn_state['right'] = curr_btn_right
            last_btn_state['left'] = curr_btn_left

            # 모니터링 출력 준비
            header = f"--- QC Capture Mode | Recorded: {len(recorded_positions)} | {datetime.datetime.now().strftime('%H:%M:%S')}"
            line_target = "" # Not applicable
            line_error = "" # Not applicable
            line_progress = f"CAPTURE | Use buttons to record | Ctrl+C to save {len(recorded_positions)} points"
            line_fault_id = f"fault_ids: {state.fault_ids}"
        # --------------------------------------------------
        # 2. Check 모드 (자동 순회) 로직
        # --------------------------------------------------
        else:
            if qc_state["test_complete"]:
                ma_input = LeaderArm.ControlInput()
                ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
                ma_input.target_torque = state.gravity_term
                return ma_input

            target_q = positions[qc_state["current_pos_idx"]]
            pos_error = np.abs(state.q_joint - target_q)
            max_error = np.max(pos_error)
            now = time.time()

            if max_error < SETTLE_THRESHOLD:
                if qc_state["settle_start"] is None:
                    qc_state["settle_start"] = now
                elif (now - qc_state["settle_start"]) >= SETTLE_DURATION:
                    qc_state["pos_reached"] = True
            else:
                qc_state["settle_start"] = None

            pos_elapsed = now - qc_state["pos_start_time"]
            pos_timed_out = pos_elapsed >= POSITION_TIMEOUT

            if qc_state["pos_reached"] or pos_timed_out:
                if pos_timed_out and not qc_state["pos_reached"]:
                    qc_state["pos_timeout_count"] += 1
                qc_state["total_pos_visited"] += 1
                qc_state["current_pos_idx"] += 1

                if qc_state["current_pos_idx"] >= num_positions:
                    qc_state["cycle_count"] += 1
                    if qc_state["cycle_count"] >= qc_state["total_cycles"]:
                        qc_state["test_complete"] = True
                    else:
                        qc_state["current_pos_idx"] = 0
                
                qc_state["settle_start"] = None
                qc_state["pos_start_time"] = now
                qc_state["pos_reached"] = False

            header = (
                f"--- QC Check Mode | Cycle {qc_state['cycle_count']+1}/{qc_state['total_cycles']}"
                f" | POS {qc_state['current_pos_idx']+1}/{num_positions} | "
                f"{datetime.datetime.now().strftime('%H:%M:%S')}"
            )
            progress_pct = (
                (qc_state["cycle_count"] * num_positions + qc_state["current_pos_idx"])
                / (qc_state["total_cycles"] * num_positions) * 100
            ) if not qc_state["test_complete"] else 100.0

            line_target  = f"target(rad):  {fmt(positions[min(qc_state['current_pos_idx'], num_positions-1)])}"
            line_error   = f"error (rad):  {fmt(pos_error)}"
            line_progress = (
                f"PROGRESS | {progress_pct:5.1f}% | Visited: {qc_state['total_pos_visited']}"
                f" | Timeouts: {qc_state['pos_timeout_count']}"
                f" | MaxErr: {max_error:.4f} rad"
            )

        # --------------------------------------------------
        # 3. 공통 통계 및 상태 출력
        # --------------------------------------------------
        current_max_streak = max(state.tool_error_counts.values()) if state.tool_error_counts else 0
        if current_max_streak > session_stats["max_streak"]:
            session_stats["max_streak"] = current_max_streak

        stats_part = f"(Tot: {session_stats['total_warnings']}, Max: {session_stats['max_streak']})"
        if state.fault_ids or state.tool_fault_ids:
            status_line = f"\033[1;31mSTATUS: [ !! CRITICAL ALARM !! - FAILED IDs: {state.fault_ids or state.tool_fault_ids} ] {stats_part}\033[0m"
        else:
            status_line = f"\033[1;32mSTATUS: [ NORMAL ] {stats_part}\033[0m"

        # Display (Only if not complete or in capture mode)
        print("\033[H\033[J", end="", flush=True)
        print(header, flush=True)
        print("-" * 60, flush=True)
        print(f"q (rad):      {fmt(state.q_joint)}", flush=True)
        if line_target: print(line_target, flush=True)
        if line_error: print(line_error, flush=True)
        print(f"current (A):  {fmt(state.current)}", flush=True)
        print(f"temp (C):     {fmt(state.temperatures)}", flush=True)
        print(f"gravity (Nm): {fmt(state.gravity_term)}", flush=True)
        print(f"BTN Status  | L: {state.button_left.button:1d} | R: {state.button_right.button:1d}", flush=True)
        print(f"fault id: {state.fault_ids}", flush=True)
        print(line_progress, flush=True)

        print("\n" + status_line, flush=True)

        # --------------------------------------------------
        # 4. Control Input 생성
        # --------------------------------------------------
        ma_input = LeaderArm.ControlInput()
        if mode == 'capture':
            # 수동 조작을 위해 중력 보상만 적용
            ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque = state.gravity_term
        else:
            # 자동 순회를 위해 위치 제어 적용
            ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentBasedPositionControlMode)
            ma_input.target_torque[:] = TORQUE_LIMIT
            ma_input.target_position[:] = positions[min(qc_state["current_pos_idx"], num_positions - 1)]

        return ma_input

    # =========================================================
    # SAFETY FUNCTION
    # =========================================================
    def safety_function(state: LeaderArm.State):
        all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR] Communication failure on IDs: {all_faults}\033[0m\n"
        print(error_msg, flush=True)
        logger.save(error_msg)

        try:
            leader_arm.DisableTorque()
            robot.power_off("12v")
            leader_arm.stop_control(torque_disable=False)
        except:
            pass
            
        print("Safety shutdown complete. Exiting.", flush=True)
        os._exit(1)

    # =========================================================
    # SIGNAL HANDLER (Ctrl+C)
    # =========================================================
    def handler(_signum, _frame):
        print("\n\nInterrupt received. Stopping...")
        if mode == 'capture':
            if recorded_positions:
                save_positions(recorded_positions)
                verify_saved_positions()
            else:
                print("[Info] No positions were recorded.")
        
        if leader_arm:
            leader_arm.close()
        try:
            robot.power_off("12v")
        except:
            pass
        print("System shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, handler)

    # =========================================================
    # START
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  Leader Arm QC Test | Mode: {mode.upper()}")
    if mode == 'check':
        print(f"  Cycles: {num_cycles} | Positions: {num_positions}")
    else:
        print(f"  Press buttons on Leader Arm to record positions.")
    print(f"{'='*60}\n")
    time.sleep(1)

    leader_arm.start_control(control, safety_function=safety_function)

    while leader_arm.ctrl_session_active:
        if mode == 'check' and qc_state["test_complete"]:
            time.sleep(2)
            break
        time.sleep(0.5)

    if mode == 'check':
        print("\n\033[1;32m[QC TEST COMPLETE] All cycles finished successfully.\033[0m")
    
    leader_arm.stop_control(torque_disable=True)
    robot.power_off("12v")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leader Arm QC Test")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument("--model", type=str, default="a", help="Robot Model Name")
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES, help="Number of test cycles")
    parser.add_argument("--mode", type=str, default="check", choices=["check", "capture"], help="check or capture mode")
    args = parser.parse_args()

    main(address=args.address, model=args.model, num_cycles=args.cycles, mode=args.mode)
