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
    leader_arm.set_max_retries(max_tool_retries=100, max_joint_retries=100)
    
    if not leader_arm.initialize(verbose=True):
        print("Failed to initialize Leader Arm")
        exit(1)
        
    if len(leader_arm.active_ids) != leader_arm.DEVICE_COUNT:
        print(f"Error: Mismatch in the number of devices detected. Expected {leader_arm.DEVICE_COUNT}, got {len(leader_arm.active_ids)}")
        exit(1)

    # 모니터링을 위한 함수
    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    def fmt_int(arr):
        return ", ".join([f"{int(x):7d}" for x in arr])


    # 사용자 정의함수
    # 1. 상태 모니터링
    # 2. 중력보상값 모터에 입력
    # 3. 통신문제(조인트가 문제발생, 혹은 버튼 트리거 관련 신호가 5번 연속으로 문제생길경우) 발생 시 12v 공급 차단
    def control(state: LeaderArm.State):
        header = f"--- Leader Arm state Monitor | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q = f"q (rad):      {fmt(state.q_joint)}"
        line_current = f"current (A):  {fmt(state.current)}"
        line_temp = f"temp (C):     {fmt(state.temperatures)}"
        line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn = f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d} | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"
        line_fault = f"Fault IDs:    {state.fault_ids}, (check time : {state.check_status_duration * 1000.0:6.1f}ms)"
        history_joints = state.fault_ids_history[:14]
        history_tools = state.fault_ids_history[14:]
        line_hist_j = f"Joint Fault count:  {fmt_int(history_joints)}"
        line_hist_t = f"Tool Fault count:   right: {int(history_tools[0]):d} | left: {int(history_tools[1]):d}"
        
        # 5. Status & Alarm Section
        if state.fault_ids:
            status_line = f"\033[1;33mSTATUS: [ WARNING - Communication Issues on IDs: {state.fault_ids} ]\033[0m"
        else:
            status_line = f"\033[1;32mSTATUS: [ NORMAL ]\033[0m"

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
        print(line_fault, flush=True)
        print(line_hist_j, flush=True)
        print(line_hist_t, flush=True)
        print("\n" + status_line, flush=True)

        # Log to file
        logger.save(f"{header}\n{line_q}\n{line_current}\n{line_temp}\n{line_torque}\n{line_grav}\n{line_btn}\n{line_fault}\n{line_hist_j}\n{line_hist_t}\n{status_line}\n")
        
        input = LeaderArm.ControlInput()
        input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        input.target_torque = state.gravity_term

        return input

    # safety_function 중복 실행 방지용 플래그
    shutdown_lock = threading.Lock()
    shutdown_done = False

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
        
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR / EXCEPTION DETECTED]\033[0m\n"
        if state.fault_ids:
            error_msg += f"Communication failure on IDs: {state.fault_ids}\n"
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
        # run_final_ping_test()
            
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
            if robot.get_control_manager_state().state == rby.ControlManagerState.State.Enabled:
                print("Disabling control manager...")
                robot.disable_control_manager()
        except Exception:
            pass
        try:
            robot.power_off("12v")
        except:
            pass
        # run_final_ping_test()
        print("System shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="19_master_arm")
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    args = parser.parse_args()

    main(address=args.address, model=args.model)
