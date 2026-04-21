import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal
import threading

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
    leader_arm.set_max_retries(max_tool_retries=5, max_joint_retries=1)
    
    if not leader_arm.initialize(verbose=True):
        print("Failed to initialize Leader Arm")
        exit(1)
        
    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print(f"Error: Mismatch in the number of devices detected. Expected {leader_arm.DEVICE_COUNT}, got {len(leader_arm.active_ids)}")
        exit(1)

    # 모니터링을 위한 함수
    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    # Session statistics for persistent monitoring
    session_stats = {
        "total_warnings": 0,
        "max_streak": 0,
        "has_warned_once": False,
        "ever_warned_ids": set()
    }

    # 사용자 정의함수
    # 1. 상태 모니터링
    # 2. 중력보상값 모터에 입력
    # 3. 통신문제(조인트가 문제발생, 혹은 버튼 트리거 관련 신호가 5번 연속으로 문제생길경우) 발생 시 12v 공급 차단
    def control(state: LeaderArm.State):
        input = LeaderArm.ControlInput()
        nonlocal session_stats
        if state.fault_ids or state.tool_fault_ids:
            print(f"Fault IDs: {state.fault_ids}")
            print(f"Tool Fault IDs: {state.tool_fault_ids}")
            input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            input.target_torque = np.zeros(14)
        else:
            # Update statistics
            if state.tool_warning_ids:
                session_stats["total_warnings"] += 1
                session_stats["has_warned_once"] = True
                for tid in state.tool_warning_ids:
                    session_stats["ever_warned_ids"].add(tid)
                
            current_max_streak = max(state.tool_error_counts.values()) if state.tool_error_counts else 0
            if current_max_streak > session_stats["max_streak"]:
                session_stats["max_streak"] = current_max_streak

            header = f"--- Leader Arm state Monitor | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
            line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
            line_q = f"q (rad):      {fmt(state.q_joint)}"
            line_current = f"current (A):  {fmt(state.current)}"
            line_temp = f"temp (C):     {fmt(state.temperatures)}"
            line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
            line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
            line_btn = f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d} | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"

            # 5. Status & Alarm Section (Fixed position at bottom)
            stats_part = f"(Tot: {session_stats['total_warnings']}, Max: {session_stats['max_streak']}, Hist IDs: {sorted(list(session_stats['ever_warned_ids']))})"
            
            if state.fault_ids or state.tool_fault_ids:
                all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
                status_line = f"\033[1;31mSTATUS: [ !! CRITICAL ALARM !! - FAILED IDs: {all_faults} ] {stats_part}\033[0m"
            elif state.tool_warning_ids:
                status_line = f"\033[1;33mSTATUS: [ WARNING - Comm jitter on IDs: {state.tool_warning_ids} ] {stats_part}\033[0m"
            elif session_stats["has_warned_once"]:
                status_line = f"\033[1;33mSTATUS: [ PAST WARNINGS DETECTED ] {stats_part}\033[0m"
            else:
                status_line = f"\033[1;32mSTATUS: [ NORMAL ] {stats_part}\033[0m"

            print("\033[H\033[J", end="", flush=True)  # Clear terminal and move cursor to top
            print(header, flush=True)
            print("-" * len(header), flush=True)
            print(line_idx, flush=True)
            print(line_q, flush=True)
            print(line_current, flush=True)
            print(line_temp, flush=True)
            print(line_torque, flush=True)
            print(line_grav, flush=True)
            print(line_btn, flush=True)
            print("\n" + status_line, flush=True)

            # Tool Warning
            if state.tool_fault_ids:
                warning_msg = f"! [TOOL WARNING] Communication failure on IDs: {state.tool_fault_ids}"
                print(warning_msg, flush=True)
                logger.save(f"{warning_msg}\n")

            # Log to file
            logger.save(f"{header}\n{line_q}\n{line_current}\n{line_temp}\n{line_torque}\n{line_grav}\n{line_btn}\n{status_line}\n")

            input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
            input.target_torque = state.gravity_term

        return input

    # safety_function 중복 실행 방지용 플래그
    shutdown_lock = threading.Lock()
    shutdown_done = False

    def run_final_ping_test():
        """부모 프로세스가 종료된 후, 별도 프로세스에서 12V 재투입 + ping test를 수행"""
        import subprocess
        import sys

        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = logger.filepath

        # 별도 프로세스에서 실행할 ping test 스크립트 (템플릿)
        ping_script = '''
import time, sys, os

SCRIPT_DIR = "__SCRIPT_DIR__"
ADDRESS = "__ADDRESS__"
MODEL = "__MODEL__"
LOG_PATH = "__LOG_PATH__"

os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import rby1_sdk as rby
from leader_arm import LeaderArm

time.sleep(2)  # 부모 프로세스가 완전히 죽고 시리얼 포트 해제될 때까지 대기

print("\\n\\033[1;36m[FINAL PING TEST] 12V 재투입 후 ping test 수행 중...\\033[0m", flush=True)
try:
    robot = rby.create_robot(ADDRESS, MODEL)
    robot.connect()
    if not robot.is_connected():
        print("\\033[1;31m[FINAL PING TEST] Robot connection failed\\033[0m", flush=True)
        sys.exit(1)

    robot.power_on("12v")
    time.sleep(0.5)

    arm = LeaderArm(control_period=0.01)
    active = arm.initialize(verbose=False)
    all_ids = list(range(arm.DOF)) + arm.tool_ids
    active = sorted(active) if active else []
    missing = sorted(set(all_ids) - set(active))

    msg = "\\n--- Final Ping Test Result ---\\n"
    msg += "  Expected IDs : " + str(sorted(all_ids)) + "\\n"
    msg += "  Active IDs   : " + str(active) + "\\n"
    msg += "  Missing IDs  : " + (str(missing) if missing else "None") + "\\n"
    if missing:
        msg += "\\033[1;31m  >> " + str(len(missing)) + "개 축 응답 없음: " + str(missing) + "\\033[0m\\n"
    else:
        msg += "\\033[1;32m  >> 모든 축 정상 응답\\033[0m\\n"
    msg += "-------------------------------\\n"

    print(msg, flush=True)

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\\n")

    robot.power_off("12v")
    print("[FINAL PING TEST] Complete. 12V powered off.", flush=True)
except Exception as e:
    print("\\033[1;31m[FINAL PING TEST] Error: " + str(e) + "\\033[0m", flush=True)
    try:
        robot.power_off("12v")
    except:
        pass
'''
        ping_script = ping_script.replace("__SCRIPT_DIR__", script_dir)
        ping_script = ping_script.replace("__ADDRESS__", address)
        ping_script = ping_script.replace("__MODEL__", model)
        ping_script = ping_script.replace("__LOG_PATH__", log_path)

        try:
            subprocess.Popen(
                [sys.executable, "-c", ping_script],
                stdout=None,   # 터미널 stdout 상속
                stderr=None,   # 터미널 stderr 상속
                stdin=subprocess.DEVNULL,
                start_new_session=True,  # 부모와 완전 분리
                cwd=script_dir
            )
            print("\033[1;36m[FINAL PING TEST] Ping test 프로세스가 백그라운드로 실행됩니다. 잠시 후 결과가 출력됩니다.\033[0m", flush=True)
        except Exception as e:
            err = f"[FINAL PING TEST] Failed to launch subprocess: {e}"
            print(f"\033[1;31m{err}\033[0m", flush=True)
            logger.save(err)

    def safety_function(state: LeaderArm.State):
        nonlocal shutdown_done
        import sys

        # 중복 실행 방지: 첫 번째 호출만 실행, 나머지는 무시
        with shutdown_lock:
            if shutdown_done:
                return
            shutdown_done = True
        
        # Flush all pending logs first
        sys.stdout.flush()
        sys.stderr.flush()
        
        all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR / EXCEPTION DETECTED]\033[0m\n"
        if all_faults:
            error_msg += f"Communication failure on IDs: {all_faults}\n"
        error_msg += "ACTION: Immediate Emergency Shutdown (Power Off 12V).\n"
        
        print(error_msg, flush=True)
        logger.save(error_msg)
        
        # Priority 1: Cleanup (제어 루프 중지, 토크 비활성화)
        if leader_arm:
            leader_arm.stop_control(torque_disable=True)

        # Priority 2: 12V 차단
        try:
            robot.power_off("12v")
        except:
            pass

        # Priority 3: 분리된 프로세스로 ping test 예약 (부모 종료 후 자동 실행됨)
        run_final_ping_test()
            
        print("Shutdown complete. Exiting.", flush=True)
        sys.stdout.flush()
        time.sleep(0.5) # Critical delay to ensure terminal displays the message
        os._exit(1)

    try:
        leader_arm.start_control(control, safety_function=safety_function)
        while leader_arm.ctrl_session_active:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping...")
    finally:
        # safety_function이 이미 처리한 경우 중복 실행 방지
        with shutdown_lock:
            if shutdown_done:
                return
            shutdown_done = True
        print("Shutting down Leader Arm engine...")
        leader_arm.stop_control(torque_disable=True)
        print("Powering off 12V...")
        try:
            robot.power_off("12v")
        except:
            pass
        run_final_ping_test()
        print("System shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="19_master_arm")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model)
