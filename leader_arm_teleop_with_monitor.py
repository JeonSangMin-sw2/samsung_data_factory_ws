"""
Teleoperation with Real-time Monitoring & Safety

마스터암으로 RBY1을 텔레오퍼레이션 하면서 실시간 상태 모니터링을 제공하고,
통신 장애 발생 시 12V 차단 safety를 적용한 코드.

References:
  - leader_arm_state_check.py : 실시간 모니터링 + safety_function
  - leader_arm_teleop.py      : 텔레오퍼레이션 로직
"""

import rby1_sdk as rby
import numpy as np
import os
import time
import logging
import argparse
import signal
import threading
import datetime
from typing import *
from dataclasses import dataclass
from leader_arm import LeaderArm

GRIPPER_DIRECTION = False

# ============================================================
# File Logger (from leader_arm_state_check.py)
# ============================================================
class File_Logger:
    def __init__(self, filepath=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, "log")
        os.makedirs(log_dir, exist_ok=True)

        if filepath is None:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.filepath = os.path.join(log_dir, f"{now_str}_teleop_monitor.txt")
        else:
            self.filepath = os.path.join(log_dir, filepath)

    def save(self, content):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(str(content) + "\n")


# ============================================================
# Data Structures & Settings (from leader_arm_teleop.py)
# ============================================================
@dataclass
class Pose:
    toros: np.typing.NDArray
    right_arm: np.typing.NDArray
    left_arm: np.typing.NDArray


class Settings:
    master_arm_loop_period = 1 / 100

    impedance_stiffness = 50
    impedance_damping_ratio = 1.0
    impedance_torque_limit = 30.0


READY_POSE = {
    "A": Pose(
        toros=np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
    ),
    "M": Pose(
        toros=np.deg2rad([0.0, 45.0, -90.0, 45.0, 0.0, 0.0]),
        right_arm=np.deg2rad([0.0, -5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
        left_arm=np.deg2rad([0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0]),
    ),
}


# ============================================================
# Gripper (from leader_arm_teleop.py)
# ============================================================
class Gripper:
    def __init__(self):
        self.bus = rby.DynamixelBus(rby.upc.GripperDeviceName)
        self.bus.open_port()
        self.bus.set_baud_rate(2_000_000)
        self.bus.set_torque_constant([1, 1])
        self.min_q = np.array([np.inf, np.inf])
        self.max_q = np.array([-np.inf, -np.inf])
        self.target_q: np.typing.NDArray = None
        self._running = False
        self._thread = None

    def initialize(self, verbose=True):
        rv = True
        for dev_id in [0, 1]:
            if not self.bus.ping(dev_id):
                if verbose:
                    logging.error(f"Dynamixel ID {dev_id} is not active")
                rv = False
            else:
                if verbose:
                    logging.info(f"Dynamixel ID {dev_id} is active")
        if rv:
            logging.info("Servo on gripper")
            self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])
        return rv

    def set_operating_mode(self, mode):
        self.bus.group_sync_write_torque_enable([(dev_id, 0) for dev_id in [0, 1]])
        self.bus.group_sync_write_operating_mode([(dev_id, mode) for dev_id in [0, 1]])
        self.bus.group_sync_write_torque_enable([(dev_id, 1) for dev_id in [0, 1]])

    def homing(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentControlMode)
        direction = 0
        q = np.array([0, 0], dtype=np.float64)
        prev_q = np.array([0, 0], dtype=np.float64)
        counter = 0
        while direction < 2:
            self.bus.group_sync_write_send_torque(
                [(dev_id, 0.5 * (1 if direction == 0 else -1)) for dev_id in [0, 1]]
            )
            rv = self.bus.group_fast_sync_read_encoder([0, 1])
            if rv is not None:
                for dev_id, enc in rv:
                    q[dev_id] = enc
            self.min_q = np.minimum(self.min_q, q)
            self.max_q = np.maximum(self.max_q, q)
            if np.array_equal(prev_q, q):
                counter += 1
            prev_q = q
            if counter >= 30:
                direction += 1
                counter = 0
            time.sleep(0.1)
        return True

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def loop(self):
        self.set_operating_mode(rby.DynamixelBus.CurrentBasedPositionControlMode)
        self.bus.group_sync_write_send_torque([(dev_id, 5) for dev_id in [0, 1]])
        while self._running:
            if self.target_q is not None:
                self.bus.group_sync_write_send_position(
                    [(dev_id, q) for dev_id, q in enumerate(self.target_q.tolist())]
                )
            time.sleep(0.1)

    def set_target(self, normalized_q):
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return
        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q


# ============================================================
# Robot Command Builders (from leader_arm_teleop.py)
# ============================================================
def joint_position_command_builder(
    pose: Pose, minimum_time, control_hold_time=0, position_mode=True
):
    right_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        right_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.right_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            right_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.right_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.right_arm))
        )

    left_arm_builder = (
        rby.JointPositionCommandBuilder()
        if position_mode
        else rby.JointImpedanceControlCommandBuilder()
    )
    (
        left_arm_builder.set_command_header(
            rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
        )
        .set_position(pose.left_arm)
        .set_minimum_time(minimum_time)
    )
    if not position_mode:
        (
            left_arm_builder.set_stiffness(
                [Settings.impedance_stiffness] * len(pose.left_arm)
            )
            .set_damping_ratio(Settings.impedance_damping_ratio)
            .set_torque_limit([Settings.impedance_torque_limit] * len(pose.left_arm))
        )

    return rby.RobotCommandBuilder().set_command(
        rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder()
            .set_torso_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(control_hold_time)
                )
                .set_position(pose.toros)
                .set_minimum_time(minimum_time)
            )
            .set_right_arm_command(right_arm_builder)
            .set_left_arm_command(left_arm_builder)
        )
    )


def move_j(robot, pose: Pose, minimum_time=5.0):
    handler = robot.send_command(joint_position_command_builder(pose, minimum_time))
    return handler.get() == rby.RobotCommandFeedback.FinishCode.Ok


# ============================================================
# Main
# ============================================================
def main(address, model_name, power, servo, control_mode):
    logger = File_Logger()

    # ===== SETUP ROBOT =====
    robot = rby.create_robot(address, model_name)
    if not robot.connect():
        logging.error(f"Failed to connect robot {address}")
        exit(1)

    supported_model = ["A", "M"]
    supported_control_mode = ["position", "impedance"]
    model = robot.model()
    dyn_model = robot.get_dynamics()
    dyn_state = dyn_model.make_state([], model.robot_joint_names)
    robot_q = None
    robot_max_q = dyn_model.get_limit_q_upper(dyn_state)
    robot_min_q = dyn_model.get_limit_q_lower(dyn_state)
    robot_max_qdot = dyn_model.get_limit_qdot_upper(dyn_state)
    robot_max_qddot = dyn_model.get_limit_qddot_upper(dyn_state)

    if control_mode == "impedance":
        robot_max_qdot[model.right_arm_idx[-1]] *= 10
        robot_max_qdot[model.left_arm_idx[-1]] *= 10

    if model.model_name not in supported_model:
        logging.error(
            f"Model {model.model_name} not supported (Supported: {supported_model})"
        )
        exit(1)
    if control_mode not in supported_control_mode:
        logging.error(
            f"Control mode {control_mode} not supported (Supported: {supported_control_mode})"
        )
        exit(1)

    position_mode = control_mode == "position"

    if not robot.is_power_on(power):
        if not robot.power_on(power):
            logging.error(f"Failed to turn power ({power}) on")
            exit(1)
    if not robot.is_servo_on(servo):
        if not robot.servo_on(servo):
            logging.error(f"Failed to servo ({servo}) on")
            exit(1)
    robot.reset_fault_control_manager()
    if not robot.enable_control_manager():
        logging.error("Failed to enable control manager")
        exit(1)
    for arm in ["right", "left"]:
        if not robot.set_tool_flange_output_voltage(arm, 12):
            logging.error(f"Failed to set tool flange output voltage ({arm}) as 12v")
            exit(1)
    robot.set_parameter("joint_position_command.cutoff_frequency", "3")
    move_j(robot, READY_POSE[model.model_name], 5)

    def robot_state_callback(state: rby.RobotState_A):
        nonlocal robot_q
        robot_q = state.position

    robot.start_state_update(robot_state_callback, 1 / Settings.master_arm_loop_period)

    # ===== SETUP GRIPPER =====
    gripper = Gripper()
    if not gripper.initialize():
        logging.error("Failed to initialize gripper")
        robot.stop_state_update()
        robot.power_off("12v")
        exit(1)
    gripper.homing()
    gripper.start()

    # ===== MASTER ARM SETUP =====
    master_arm = LeaderArm(
        control_period=Settings.master_arm_loop_period,
        check_temp=True,
        check_bus=True,
        check_transform=True,
    )
    active_ids = master_arm.initialize(verbose=True)

    if len(master_arm.active_ids) != master_arm.DEVICE_COUNT:
        logging.error(
            f"Mismatch in the number of devices detected. "
            f"Expected {master_arm.DEVICE_COUNT}, got {len(master_arm.active_ids)}"
        )
        exit(1)

    # ===== TELEOP PARAMETERS =====
    ma_q_limit_barrier = 0.5
    ma_min_q = np.deg2rad(
        [-360, -30, 0, -135, -90, 35, -360, -360, 10, -90, -135, -90, 35, -360]
    )
    ma_max_q = np.deg2rad(
        [360, -10, 90, -60, 90, 80, 360, 360, 30, 0, -60, 90, 80, 360]
    )
    ma_torque_limit = np.array([3.5, 3.5, 3.5, 1.5, 1.5, 1.5, 1.5] * 2)
    ma_viscous_gain = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.002] * 2)
    right_q = None
    left_q = None
    right_minimum_time = 1.0
    left_minimum_time = 1.0

    stream = robot.create_command_stream(priority=1)
    stream.send_command(
        joint_position_command_builder(
            READY_POSE[model.model_name],
            minimum_time=5,
            control_hold_time=1e6,
            position_mode=position_mode,
        )
    )

    # ===== SESSION STATISTICS (from state_check) =====
    session_stats = {
        "total_warnings": 0,
        "max_streak": 0,
        "has_warned_once": False,
        "ever_warned_ids": set(),
    }

    def fmt(arr):
        return ", ".join([f"{x:7.3f}" for x in arr])

    # =========================================================
    # CONTROL CALLBACK (Teleop + Real-time Monitoring)
    # =========================================================
    def master_arm_control_loop(state: LeaderArm.State):
        nonlocal position_mode, right_q, left_q
        nonlocal right_minimum_time, left_minimum_time
        nonlocal session_stats

        if right_q is None:
            right_q = state.q_joint[0:7]
        if left_q is None:
            left_q = state.q_joint[7:14]

        # --------------------------------------------------
        # 1. Real-time Monitoring Display (from state_check)
        # --------------------------------------------------
        # Update session statistics
        if state.tool_warning_ids:
            session_stats["total_warnings"] += 1
            session_stats["has_warned_once"] = True
            for tid in state.tool_warning_ids:
                session_stats["ever_warned_ids"].add(tid)

        current_max_streak = max(state.tool_error_counts.values()) if state.tool_error_counts else 0
        if current_max_streak > session_stats["max_streak"]:
            session_stats["max_streak"] = current_max_streak

        header = f"--- Teleop Monitor | {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]} ---"
        line_idx = "index:        " + ", ".join([f"{i:7d}" for i in range(len(state.q_joint))])
        line_q = f"q (rad):      {fmt(state.q_joint)}"
        line_current = f"current (A):  {fmt(state.current)}"
        line_temp = f"temp (C):     {fmt(state.temperatures)}"
        line_torque = f"torque (Nm):  {fmt(state.torque_joint)}"
        line_grav = f"gravity (Nm): {fmt(state.gravity_term)}"
        line_btn = (
            f"BTN   | L: {state.button_left.button:1d} TRG: {state.button_left.trigger:4d}"
            f" | R: {state.button_right.button:1d} TRG: {state.button_right.trigger:4d}"
        )
        line_teleop = (
            f"TELEOP | R_btn: {state.button_right.button} L_btn: {state.button_left.button}"
            f" | mode: {'position' if position_mode else 'impedance'}"
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
        print(line_current, flush=True)
        print(line_temp, flush=True)
        print(line_torque, flush=True)
        print(line_grav, flush=True)
        print(line_btn, flush=True)
        print(line_teleop, flush=True)
        print("\n" + status_line, flush=True)

        if state.tool_fault_ids:
            warning_msg = f"! [TOOL WARNING] Communication failure on IDs: {state.tool_fault_ids}"
            print(warning_msg, flush=True)
            logger.save(warning_msg)

        # Log to file
        logger.save(
            f"{header}\n{line_q}\n{line_current}\n{line_temp}\n{line_torque}\n"
            f"{line_grav}\n{line_btn}\n{line_teleop}\n{status_line}"
        )

        # --------------------------------------------------
        # 2. Gripper Control
        # --------------------------------------------------
        gripper.set_target(
            np.array(
                [state.button_right.trigger / 1000, state.button_left.trigger / 1000]
            )
        )

        # --------------------------------------------------
        # 3. Master Arm Torque Calculation (from teleop)
        # --------------------------------------------------
        ma_input = LeaderArm.ControlInput()

        torque = (
            state.gravity_term * 1.0
            + ma_q_limit_barrier
            * (
                np.maximum(ma_min_q - state.q_joint, 0)
                + np.minimum(ma_max_q - state.q_joint, 0)
            )
            + ma_viscous_gain * state.qvel_joint
        )
        torque = np.clip(torque, -ma_torque_limit, ma_torque_limit)

        # Right arm
        if state.button_right.button == 1:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[0:7] = torque[0:7] * 0.6
            right_q = state.q_joint[0:7]
        else:
            ma_input.target_operating_mode[0:7].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[0:7] = ma_torque_limit[0:7]
            ma_input.target_position[0:7] = right_q

        # Left arm
        if state.button_left.button == 1:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentControlMode
            )
            ma_input.target_torque[7:14] = torque[7:14] * 0.6
            left_q = state.q_joint[7:14]
        else:
            ma_input.target_operating_mode[7:14].fill(
                rby.DynamixelBus.CurrentBasedPositionControlMode
            )
            ma_input.target_torque[7:14] = ma_torque_limit[7:14]
            ma_input.target_position[7:14] = left_q

        # --------------------------------------------------
        # 4. Build & Send Robot Command (from teleop)
        # --------------------------------------------------
        q = robot_q.copy() if robot_q is not None else np.zeros(len(model.robot_joint_names))
        q[model.right_arm_idx] = right_q
        q[model.left_arm_idx] = left_q
        dyn_state.set_q(q)
        dyn_model.compute_forward_kinematics(dyn_state)

        rc = rby.BodyComponentBasedCommandBuilder()

        if state.button_right.button:
            right_minimum_time -= Settings.master_arm_loop_period
            right_minimum_time = max(
                right_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            right_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                right_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        right_q,
                        robot_min_q[model.right_arm_idx],
                        robot_max_q[model.right_arm_idx],
                    )
                )
                .set_velocity_limit(robot_max_qdot[model.right_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.right_arm_idx] * 30)
                .set_minimum_time(right_minimum_time)
            )
            if not position_mode:
                (
                    right_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(model.right_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(model.right_arm_idx)
                    )
                )
            rc.set_right_arm_command(right_arm_builder)
        else:
            right_minimum_time = 0.8

        if state.button_left.button:
            left_minimum_time -= Settings.master_arm_loop_period
            left_minimum_time = max(
                left_minimum_time, Settings.master_arm_loop_period * 1.01
            )
            left_arm_builder = (
                rby.JointPositionCommandBuilder()
                if position_mode
                else rby.JointImpedanceControlCommandBuilder()
            )
            (
                left_arm_builder.set_command_header(
                    rby.CommandHeaderBuilder().set_control_hold_time(1e6)
                )
                .set_position(
                    np.clip(
                        left_q,
                        robot_min_q[model.left_arm_idx],
                        robot_max_q[model.left_arm_idx],
                    )
                )
                .set_velocity_limit(robot_max_qdot[model.left_arm_idx])
                .set_acceleration_limit(robot_max_qddot[model.left_arm_idx] * 30)
                .set_minimum_time(left_minimum_time)
            )
            if not position_mode:
                (
                    left_arm_builder.set_stiffness(
                        [Settings.impedance_stiffness] * len(model.left_arm_idx)
                    )
                    .set_damping_ratio(Settings.impedance_damping_ratio)
                    .set_torque_limit(
                        [Settings.impedance_torque_limit] * len(model.left_arm_idx)
                    )
                )
            rc.set_left_arm_command(left_arm_builder)
        else:
            left_minimum_time = 0.8

        stream.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(rc)
            )
        )

        return ma_input

    # =========================================================
    # SAFETY FUNCTION (from leader_arm_state_check.py)
    # 통신 장애 시 즉시 12V 차단 후 프로세스 종료
    # =========================================================
    def safety_function(state: LeaderArm.State):
        import sys

        sys.stdout.flush()
        sys.stderr.flush()

        all_faults = sorted(list(state.fault_ids) + list(state.tool_fault_ids))
        error_msg = f"\n\n\033[1;31m[CRITICAL ERROR / EXCEPTION DETECTED]\033[0m\n"
        if all_faults:
            error_msg += f"Communication failure on IDs: {all_faults}\n"
        error_msg += "ACTION: Immediate Emergency Shutdown (Power Off 12V).\n"

        print(error_msg, flush=True)
        logger.save(error_msg)

        # Priority 1: Hardware Safety — 마스터암 토크 즉시 끄기 (stop_control은 데드락 위험이므로 직접 호출)
        try:
            master_arm.DisableTorque()
        except Exception:
            pass

        # Priority 2: 12V 즉시 차단
        try:
            robot.power_off("12v")
        except Exception:
            pass

        # Priority 3: Robot control 정리
        try:
            robot.stop_state_update()
        except Exception:
            pass
        try:
            robot.cancel_control()
        except Exception:
            pass
        try:
            robot.disable_control_manager()
        except Exception:
            pass

        # Priority 4: Master arm & gripper cleanup
        try:
            master_arm.stop_control(torque_disable=False)  # 토크는 이미 끔
        except Exception:
            pass
        try:
            gripper.stop()
        except Exception:
            pass

        print("Shutdown complete. Exiting.", flush=True)
        sys.stdout.flush()
        time.sleep(0.5)
        os._exit(1)

    # =========================================================
    # START CONTROL LOOP
    # =========================================================
    master_arm.start_control(master_arm_control_loop, safety_function=safety_function)

    # ===== SIGNAL HANDLER (Ctrl+C) =====
    def handler(signum, frame):
        print("\n\nInterrupt received. Stopping...")
        robot.stop_state_update()
        if master_arm:
            master_arm.close()
        robot.cancel_control()
        time.sleep(0.5)
        robot.disable_control_manager()
        robot.power_off("12v")
        gripper.stop()
        print("System shutdown complete.")
        os._exit(0)

    signal.signal(signal.SIGINT, handler)

    # Main thread waits
    while master_arm.ctrl_session_active:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teleoperation with Real-time Monitoring & Safety"
    )
    parser.add_argument("--address", type=str, required=True, help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Regex pattern for power device names (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default="torso_.*|right_arm_.*|left_arm_.*",
        help="Regex pattern for servo names",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="position",
        choices=["position", "impedance"],
        help="Control mode: 'position' or 'impedance' (default: 'position')",
    )
    args = parser.parse_args()

    main(
        address=args.address,
        model_name=args.model,
        power=args.power,
        servo=args.servo,
        control_mode=args.mode,
    )
