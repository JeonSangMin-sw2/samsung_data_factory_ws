"""
Leader Arm QC Test

리더암이 스스로 N 번의 사이클 동안 미리 정의된 웨이포인트를 순회하며 움직입니다.
움직이는 과정에서 연결 상태를 지속적으로 체크하는 Quality Control 테스트 코드입니다.

주요 기능:
  - 실시간 상태 모니터링 (teleop_with_leader_arm.py 참고)
  - ~10개의 웨이포인트를 Position 모드(CurrentBasedPositionControlMode)로 순회
  - 로봇 쪽으로 제어 명령을 보내지 않음 (리더암만 움직임)
  - 통신 장애 발생 시 12V 차단 safety 적용

References:
  - teleop_with_leader_arm.py : 실시간 모니터링 + position 모드 제어
  - leader_arm_state_check.py : safety_function 패턴
"""

import os
import sys
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
SETTLE_THRESHOLD = 0.10       # rad (~5.7 deg) — 웨이포인트 도달 판정 임계값
SETTLE_DURATION = 0.5         # sec — 임계값 이내로 유지해야 하는 시간
WAYPOINT_TIMEOUT = 8.0        # sec — 한 웨이포인트에서 대기하는 최대 시간
TORQUE_LIMIT = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)


# ============================================================
# 웨이포인트 정의 (~10개)
# 리더암 조인트 범위 내에서 안전한 포인트들을 정의
#   Right arm (0-6):  J0[-360,360], J1[-30,-10], J2[0,90], J3[-135,-60],
#                     J4[-90,90],   J5[35,80],   J6[-360,360]
#   Left arm  (7-13): J7[-360,360], J8[10,30],   J9[-90,0], J10[-135,-60],
#                     J11[-90,90],  J12[35,80],  J13[-360,360]
# ============================================================
WAYPOINTS = np.deg2rad([
    # Waypoint 0: Home / Neutral
    [   0, -20,  45,  -90,   0,  57,   0,    0,  20, -45,  -90,   0,  57,   0],
    # Waypoint 1: 양팔 엘보우 굽힘
    [   0, -20,  45, -120,   0,  57,   0,    0,  20, -45, -120,   0,  57,   0],
    # Waypoint 2: 양팔 엘보우 펼침
    [   0, -20,  45,  -70,   0,  57,   0,    0,  20, -45,  -70,   0,  57,   0],
    # Waypoint 3: 손목 피치 변경
    [   0, -20,  45,  -90,   0,  70,   0,    0,  20, -45,  -90,   0,  70,   0],
    # Waypoint 4: 손목 피치 + 엘보우
    [   0, -20,  45, -110,   0,  40,   0,    0,  20, -45, -110,   0,  40,   0],
    # Waypoint 5: 어깨 롤 변경
    [   0, -15,  45,  -90,   0,  57,   0,    0,  15, -45,  -90,   0,  57,   0],
    # Waypoint 6: 어깨 요 + 엘보우 변경
    [   0, -20,  70, -100,   0,  57,   0,    0,  20, -70, -100,   0,  57,   0],
    # Waypoint 7: 손목 요1 변화
    [   0, -20,  45,  -90,  45,  57,   0,    0,  20, -45,  -90, -45,  57,   0],
    # Waypoint 8: 손목 요1 반대
    [   0, -20,  45,  -90, -45,  57,   0,    0,  20, -45,  -90,  45,  57,   0],
    # Waypoint 9: 복합 동작
    [   0, -25,  60, -110,  30,  65,   0,    0,  25, -60, -110, -30,  65,   0],
])


# ============================================================
# File Logger
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


# ============================================================
# Main
# ============================================================
def main(address, model, num_cycles):
    logger = FileLogger()

    # ===== SETUP ROBOT (12V 공급만 사용, 로봇 제어 명령 없음) =====
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
    waypoints = WAYPOINTS
    num_waypoints = len(waypoints)

    qc_state = {
        "current_wp_idx": 0,
        "cycle_count": 0,
        "total_cycles": num_cycles,
        "settle_start": None,        # 도달 판정 시작 시간
        "wp_start_time": time.time(), # 현재 웨이포인트 시작 시간
        "wp_reached": False,
        "test_complete": False,
        "wp_timeout_count": 0,        # 타임아웃으로 넘어간 웨이포인트 수
        "total_wp_visited": 0,        # 총 방문한 웨이포인트 수
    }

    # Session statistics (from teleop_with_leader_arm.py)
    session_stats = {
        "total_warnings": 0,
        "max_streak": 0,
        "has_warned_once": False,
        "ever_warned_ids": set(),
    }

    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    # =========================================================
    # CONTROL CALLBACK
    # =========================================================
    def control(state: LeaderArm.State):
        nonlocal session_stats, qc_state

        # --------------------------------------------------
        # 0. 테스트 완료 시 gravity comp만 유지
        # --------------------------------------------------
        if qc_state["test_complete"]:
            ma_input = LeaderArm.ControlInput()
            ma_input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            ma_input.target_torque = state.gravity_term
            return ma_input

        # --------------------------------------------------
        # 1. Session Statistics 업데이트
        # --------------------------------------------------
        if state.tool_warning_ids:
            session_stats["total_warnings"] += 1
            session_stats["has_warned_once"] = True
            for tid in state.tool_warning_ids:
                session_stats["ever_warned_ids"].add(tid)

        current_max_streak = (
            max(state.tool_error_counts.values()) if state.tool_error_counts else 0
        )
        if current_max_streak > session_stats["max_streak"]:
            session_stats["max_streak"] = current_max_streak

        # --------------------------------------------------
        # 2. 웨이포인트 도달 판정 및 전환
        # --------------------------------------------------
        target_q = waypoints[qc_state["current_wp_idx"]]
        pos_error = np.abs(state.q_joint - target_q)
        max_error = np.max(pos_error)
        now = time.time()

        if max_error < SETTLE_THRESHOLD:
            if qc_state["settle_start"] is None:
                qc_state["settle_start"] = now
            elif (now - qc_state["settle_start"]) >= SETTLE_DURATION:
                qc_state["wp_reached"] = True
        else:
            qc_state["settle_start"] = None

        # 타임아웃 체크
        wp_elapsed = now - qc_state["wp_start_time"]
        wp_timed_out = wp_elapsed >= WAYPOINT_TIMEOUT

        # 다음 웨이포인트로 전환
        if qc_state["wp_reached"] or wp_timed_out:
            if wp_timed_out and not qc_state["wp_reached"]:
                qc_state["wp_timeout_count"] += 1

            qc_state["total_wp_visited"] += 1
            qc_state["current_wp_idx"] += 1

            # 사이클 완료 체크
            if qc_state["current_wp_idx"] >= num_waypoints:
                qc_state["cycle_count"] += 1
                if qc_state["cycle_count"] >= qc_state["total_cycles"]:
                    qc_state["test_complete"] = True
                else:
                    qc_state["current_wp_idx"] = 0

            # 상태 리셋
            qc_state["settle_start"] = None
            qc_state["wp_start_time"] = now
            qc_state["wp_reached"] = False

        # --------------------------------------------------
        # 3. 실시간 모니터링 디스플레이
        # --------------------------------------------------
        header = (
            f"--- QC Test Monitor | Cycle {qc_state['cycle_count']+1}/{qc_state['total_cycles']}"
            f" | WP {qc_state['current_wp_idx']+1}/{num_waypoints}"
            f" | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        )
        line_idx     = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q       = f"q (rad):      {fmt(state.q_joint)}"
        line_target  = f"target(rad):  {fmt(waypoints[min(qc_state['current_wp_idx'], num_waypoints-1)])}"
        line_error   = f"error (rad):  {fmt(pos_error)}"
        line_current = f"current (A):  {fmt(state.current)}"
        line_temp    = f"temp (C):     {fmt(state.temperatures)}"
        line_torque  = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav    = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn     = (
            f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d}"
            f" | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"
        )

        progress_pct = (
            (qc_state["cycle_count"] * num_waypoints + qc_state["current_wp_idx"])
            / (qc_state["total_cycles"] * num_waypoints) * 100
        ) if not qc_state["test_complete"] else 100.0

        line_progress = (
            f"PROGRESS | {progress_pct:5.1f}% | Visited: {qc_state['total_wp_visited']}"
            f" | Timeouts: {qc_state['wp_timeout_count']}"
            f" | MaxErr: {max_error:.4f} rad"
            f" | WP elapsed: {wp_elapsed:.1f}s"
        )

        # Status line
        stats_part = (
            f"(Tot: {session_stats['total_warnings']}, "
            f"Max: {session_stats['max_streak']}, "
            f"Hist IDs: {sorted(list(session_stats['ever_warned_ids']))})"
        )

        if state.fault_ids or state.tool_fault_ids:
            all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
            status_line = f"\033[1;31mSTATUS: [ !! CRITICAL ALARM !! - FAILED IDs: {all_faults} ] {stats_part}\033[0m"
        elif state.tool_warning_ids:
            status_line = f"\033[1;33mSTATUS: [ WARNING - Comm jitter on IDs: {state.tool_warning_ids} ] {stats_part}\033[0m"
        elif session_stats["has_warned_once"]:
            status_line = f"\033[1;33mSTATUS: [ PAST WARNINGS DETECTED ] {stats_part}\033[0m"
        else:
            status_line = f"\033[1;32mSTATUS: [ NORMAL ] {stats_part}\033[0m"

        # Display
        print("\033[H\033[J", end="", flush=True)
        print(header, flush=True)
        print("-" * len(header), flush=True)
        print(line_idx, flush=True)
        print(line_q, flush=True)
        print(line_target, flush=True)
        print(line_error, flush=True)
        print(line_current, flush=True)
        print(line_temp, flush=True)
        print(line_torque, flush=True)
        print(line_grav, flush=True)
        print(line_btn, flush=True)
        print(line_progress, flush=True)
        print("\n" + status_line, flush=True)

        if state.tool_fault_ids:
            warning_msg = f"! [TOOL WARNING] Communication failure on IDs: {state.tool_fault_ids}"
            print(warning_msg, flush=True)
            logger.save(warning_msg)

        if qc_state["test_complete"]:
            print("\n\033[1;32m[QC TEST COMPLETE] All cycles finished successfully.\033[0m", flush=True)

        # Log to file
        logger.save(
            f"{header}\n{line_q}\n{line_target}\n{line_error}\n"
            f"{line_current}\n{line_temp}\n{line_torque}\n{line_grav}\n"
            f"{line_btn}\n{line_progress}\n{status_line}"
        )

        # --------------------------------------------------
        # 4. Position 명령 생성
        # --------------------------------------------------
        ma_input = LeaderArm.ControlInput()

        # CurrentBasedPositionControlMode로 전 조인트 설정
        ma_input.target_operating_mode.fill(
            rby.DynamixelBus.CurrentBasedPositionControlMode
        )
        ma_input.target_torque[:] = TORQUE_LIMIT
        ma_input.target_position[:] = waypoints[
            min(qc_state["current_wp_idx"], num_waypoints - 1)
        ]

        return ma_input

    # =========================================================
    # SAFETY FUNCTION (통신 장애 시 즉시 12V 차단)
    # =========================================================
    def safety_function(state: LeaderArm.State):
        sys.stdout.flush()
        sys.stderr.flush()

        all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR / EXCEPTION DETECTED]\033[0m\n"
        if all_faults:
            error_msg += f"Communication failure on IDs: {all_faults}\n"
        error_msg += "ACTION: Immediate Emergency Shutdown (Power Off 12V).\n"

        print(error_msg, flush=True)
        logger.save(error_msg)

        # QC 결과 요약 기록
        summary = (
            f"\n[QC TEST ABORTED] Cycle {qc_state['cycle_count']+1}/{qc_state['total_cycles']}, "
            f"WP {qc_state['current_wp_idx']+1}/{num_waypoints}\n"
            f"Visited: {qc_state['total_wp_visited']}, Timeouts: {qc_state['wp_timeout_count']}\n"
        )
        print(summary, flush=True)
        logger.save(summary)

        # Priority 1: 마스터암 토크 즉시 끄기
        try:
            leader_arm.DisableTorque()
        except Exception:
            pass

        # Priority 2: 12V 즉시 차단
        try:
            robot.power_off("12v")
        except Exception:
            pass

        # Priority 3: Cleanup
        try:
            leader_arm.stop_control(torque_disable=False)
        except Exception:
            pass

        print("Shutdown complete. Exiting.", flush=True)
        sys.stdout.flush()
        time.sleep(0.5)
        os._exit(1)

    # =========================================================
    # START
    # =========================================================
    print(f"\n{'='*60}")
    print(f"  Leader Arm QC Test")
    print(f"  Cycles: {num_cycles} | Waypoints per cycle: {num_waypoints}")
    print(f"  Settle threshold: {SETTLE_THRESHOLD:.3f} rad ({np.rad2deg(SETTLE_THRESHOLD):.1f} deg)")
    print(f"  Settle duration: {SETTLE_DURATION:.1f}s | Timeout: {WAYPOINT_TIMEOUT:.1f}s")
    print(f"{'='*60}\n")
    time.sleep(2)

    leader_arm.start_control(control, safety_function=safety_function)

    # ===== SIGNAL HANDLER (Ctrl+C) =====
    def handler(signum, frame):
        print("\n\nInterrupt received. Stopping...")
        summary = (
            f"\n[QC TEST INTERRUPTED] Cycle {qc_state['cycle_count']+1}/{qc_state['total_cycles']}, "
            f"WP {qc_state['current_wp_idx']+1}/{num_waypoints}\n"
            f"Visited: {qc_state['total_wp_visited']}, Timeouts: {qc_state['wp_timeout_count']}\n"
        )
        print(summary)
        logger.save(summary)

        if leader_arm:
            leader_arm.close()
        try:
            robot.power_off("12v")
        except Exception:
            pass
        print("System shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, handler)

    # Main thread: 테스트 완료까지 대기
    while leader_arm.ctrl_session_active:
        if qc_state["test_complete"]:
            # 테스트 완료 후 잠시 대기 (마지막 모니터링 표시용)
            time.sleep(3)
            break
        time.sleep(1)

    # ===== 테스트 완료 후 정리 =====
    summary = (
        f"\n{'='*60}\n"
        f"  QC Test Complete\n"
        f"  Cycles completed: {qc_state['cycle_count']}/{qc_state['total_cycles']}\n"
        f"  Total waypoints visited: {qc_state['total_wp_visited']}\n"
        f"  Waypoint timeouts: {qc_state['wp_timeout_count']}\n"
        f"  Total comm warnings: {session_stats['total_warnings']}\n"
        f"  Warning IDs history: {sorted(list(session_stats['ever_warned_ids']))}\n"
        f"{'='*60}\n"
    )
    print(summary)
    logger.save(summary)

    print("Shutting down Leader Arm...")
    leader_arm.stop_control(torque_disable=True)
    print("Powering off 12V...")
    try:
        robot.power_off("12v")
    except Exception:
        pass
    print("System shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Leader Arm QC Test — 리더암 자동 순회 품질검사"
    )
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--cycles", type=int, default=DEFAULT_CYCLES,
        help=f"Number of test cycles (default: {DEFAULT_CYCLES})"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model, num_cycles=args.cycles)
