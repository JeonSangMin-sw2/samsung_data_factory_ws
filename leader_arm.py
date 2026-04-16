import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal
import threading
import logging
import copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "models", "model.urdf")
LEADER_ARM_DEVICE_NAME = rby.upc.MasterArmDeviceName

import queue

GRIPPER_DIRECTION = True

class LeaderArm:
    DOF = 14
    DEVICE_COUNT = 16
    RIGHT_TOOL_ID = 0x80
    LEFT_TOOL_ID = 0x81
    MAXIMUM_TORQUE = 0.5
    TORQUE_SCALING = 0.5
    kBaseLinkId = 0
    kRightLinkId = 7
    kLeftLinkId = 14

    class ButtonSnapshot:
        __slots__ = ['button', 'trigger']
        def __init__(self, b, t):
            self.button = b
            self.trigger = t

    class State:
        __slots__ = [
            'q_joint', 'qvel_joint', 'torque_joint', 'gravity_term', 
            'operating_mode', 'target_position', 'button_right', 
            'button_left', 'T_right', 'T_left', 'temperatures',
            'fault_ids', 'tool_fault_ids', 'tool_warning_ids', 'current',
            'tool_error_counts'
        ]
        def __init__(self, dof=14):
            self.q_joint = np.zeros(dof, dtype=np.float64)
            self.qvel_joint = np.zeros(dof, dtype=np.float64)
            self.torque_joint = np.zeros(dof, dtype=np.float64)
            self.gravity_term = np.zeros(dof, dtype=np.float64)
            self.operating_mode = np.full(dof, -1, dtype=np.int64)
            self.target_position = np.zeros(dof, dtype=np.float64)
            self.button_right = rby.DynamixelBus.ButtonState()
            self.button_left = rby.DynamixelBus.ButtonState()
            self.T_right = np.eye(4)
            self.T_left = np.eye(4)
            self.temperatures = np.zeros(dof, dtype=np.float64)
            self.fault_ids = []
            self.tool_fault_ids = []
            self.tool_warning_ids = []
            self.current = np.zeros(dof, dtype=np.float64)
            self.tool_error_counts = {}

        def copy(self):
            # Create a shallow copy of the object structure
            snapshot = copy.copy(self)
            
            # Re-allocate unique arrays for the snapshot to prevent thread race conditions
            dof = len(self.q_joint)
            snapshot.q_joint = np.zeros(dof, dtype=np.float64)
            snapshot.qvel_joint = np.zeros(dof, dtype=np.float64)
            snapshot.torque_joint = np.zeros(dof, dtype=np.float64)
            snapshot.gravity_term = np.zeros(dof, dtype=np.float64)
            snapshot.operating_mode = np.full(dof, -1, dtype=np.int64)
            snapshot.target_position = np.zeros(dof, dtype=np.float64)
            snapshot.temperatures = np.zeros(dof, dtype=np.float64)
            snapshot.fault_ids = []
            snapshot.tool_fault_ids = []
            snapshot.tool_warning_ids = []
            snapshot.current = np.zeros(dof, dtype=np.float64)
            snapshot.tool_error_counts = {}
            
            # Copy data into new arrays
            self.copy_to(snapshot)
            return snapshot

        def copy_to(self, target):
            """Efficiently copies data into an existing State object to avoid allocations."""
            target.q_joint[:] = self.q_joint
            target.qvel_joint[:] = self.qvel_joint
            target.torque_joint[:] = self.torque_joint
            target.gravity_term[:] = self.gravity_term
            target.operating_mode[:] = self.operating_mode
            target.target_position[:] = self.target_position
            target.temperatures[:] = self.temperatures
            target.fault_ids = list(self.fault_ids)
            target.tool_fault_ids = list(self.tool_fault_ids)
            target.tool_warning_ids = list(self.tool_warning_ids)
            target.current[:] = self.current
            target.tool_error_counts = dict(self.tool_error_counts)

            # Handle button snapshots (always create a new frozen snapshot for the state)
            target.button_right = LeaderArm.ButtonSnapshot(self.button_right.button, self.button_right.trigger)
            target.button_left = LeaderArm.ButtonSnapshot(self.button_left.button, self.button_left.trigger)

            # Transformation matrices (4x4 numpy arrays)
            if target.T_right is not None:
                target.T_right[:] = self.T_right
            else:
                target.T_right = self.T_right.copy()
            
            if target.T_left is not None:
                target.T_left[:] = self.T_left
            else:
                target.T_left = self.T_left.copy()

    class ControlInput:
        def __init__(self, dof=14):
            self.target_operating_mode = np.full(dof, -1, dtype=int)
            self.target_position = np.zeros(dof, dtype=np.float64)
            self.target_torque = np.zeros(dof, dtype=np.float64)
    
    class EventLoop:
        def __init__(self):
            self._tasks = queue.Queue()
            self._running = False
            self._paused = True
            self._thread = None
            self._lock = threading.Lock()

        def start(self):
            with self._lock:
                if self._thread is None or not self._thread.is_alive():
                    self._running = True
                    self._paused = False
                    self._thread = threading.Thread(target=self._worker, daemon=True)
                    self._thread.start()

        def stop(self):
            self._running = False
            self._tasks.put(None)  # Wake up worker
            if self._thread:
                self._thread.join()
                self._thread = None

        def pause(self):
            self._paused = True

        def unpause(self):
            self._paused = False

        def push_task(self, task):
            self._tasks.put(task)

        def push_cyclic_task(self, task, period_sec):
            def cyclic_wrapper():
                if not self._running:
                    return
                
                start_time = time.time()
                task()
                
                if not self._running:
                    return

                elapsed = time.time() - start_time
                sleep_time = period_sec - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                if self._running:
                    self.push_task(cyclic_wrapper)

            self.push_task(cyclic_wrapper)

        def _worker(self):
            while self._running:
                task = self._tasks.get()
                if task is None or not self._running:
                    break
                if self._paused:
                    self._tasks.task_done()
                    continue
                try:
                    task()
                except Exception as e:
                    logging.error(f"Error in EventLoop task: {e}")
                finally:
                    self._tasks.task_done()

    # LeaderArm class init function
    def __init__(self, dev_name=LEADER_ARM_DEVICE_NAME, control_period=0.01, check_temp=True, check_bus=True, check_transform=True):
        self.dev_name = dev_name
        self.bus = rby.DynamixelBus(dev_name)
        self.ev = self.EventLoop()
        self.ctrl_ev = self.EventLoop()
        self.control_period = control_period
        self.ctrl_session_active = False # Tracks if start_control is active
        self.ctrl_callback_busy = False   # Tracks if the callback is currently executing
        self.control_callback = None
        self.safety_function = None
        self.temp_flag = check_temp
        self.bus_flag = check_bus
        self.transform_flag = check_transform

        self.torque_constant = np.array([1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043,
                                         1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043])
        self.active_ids = []
        self.motor_ids = list(range(self.DOF))
        self.tool_ids = [self.RIGHT_TOOL_ID, self.LEFT_TOOL_ID]
        self.active_joint_ids = [] # Dynamically set during initialize
        self.active_tool_ids = []  # Dynamically set during initialize

        self.initialized = False
        self.operating_mode_init = False
        self.state = self.State(self.DOF)
        self.control_callback = None
        self.model_path = URDF_PATH
        self.is_running = False
        self.tool_error_counts = {tid: 0 for tid in self.tool_ids}
        self.MAX_TOOL_RETRIES = 5 # 10 consecutive fails (~0.1s at 100Hz)

    
    def SetControlPeriod(self, control_period):
        self.control_period = control_period
    
    def check_temperature(self, enable: bool):
        self.temp_flag = enable

    def check_bus_state(self, enable: bool):
        self.bus_flag = enable

    def check_transform_state(self, enable: bool):
        self.transform_flag = enable

    def SetModelPath(self, model_path):
        self.model_path = model_path
        if self.is_running:
            self._init_dynamics()

    def _init_dynamics(self, model_name, urdf_path, gravity=None):
        if gravity is None:
            gravity = [0, 0, -9.81, 0, 0, 0]
        self.robot = rby.make_state(model_name, urdf_path, gravity)
        self.dyn_state = self.robot.make_state(
            ["Base", "Link_0R", "Link_1R", "Link_2R", "Link_3R", "Link_4R", "Link_5R", "Link_6R", "Link_0L", "Link_1L",
             "Link_2L", "Link_3L", "Link_4L", "Link_5L", "Link_6L"],
            ["J0_Shoulder_Pitch_R", "J1_Shoulder_Roll_R", "J2_Shoulder_Yaw_R", "J3_Elbow_R", "J4_Wrist_Yaw1_R",
             "J5_Wrist_Pitch_R", "J6_Wrist_Yaw2_R", "J7_Shoulder_Pitch_L", "J8_Shoulder_Roll_L", "J9_Shoulder_Yaw_L",
             "J10_Elbow_L", "J11_Wrist_Yaw1_L", "J12_Wrist_Pitch_L", "J13_Wrist_Yaw2_L"]
        )

    def SetTorqueConstant(self, torque_constant):
        self.torque_constant = np.array(torque_constant)
        if self.initialized:
            self.bus.set_torque_constant(self.torque_constant.tolist())

    def initialize(self, verbose=True, gravity=None):
        """
        Initialize the arm.
        - Section 1: Setup motors and bus
        - Section 2: Identify Active Motors (Joints vs Tools)
        - Section 3: Initialize Dynamics Model (URDF)
        """
        logging.basicConfig(level=logging.INFO)
        
        # 1. Setup Motors
        self._init_motors()
        
        # 2. Open Bus
        if not self.bus.open_port():
            logging.error("[LeaderArm] Port Open Failed")
            return False
        self.bus_flag = True
        
        # 3. Identify and Set Gains
        # ... existing logic ...
        self.active_ids = self.bus.scan(253)
        self.active_joint_ids = [mid for mid in self.active_ids if mid < self.DOF]
        self.active_tool_ids = [mid for mid in self.active_ids if mid >= 0x80]
        
        # 4. Initialize Dynamics
        try:
            self._init_dynamics(self.model_name, self.urdf_path, gravity=gravity)
        except Exception as e:
            logging.error(f"[LeaderArm] Dynamics Init Failed: {e}")
            return False
            
        return True

    def check_motor_status(self, verbose=True):
        active_ids = []
        # Check Motor 0~13 and Tool Motor 0x80, 0x81
        self.motor_ids = list(range(self.DOF))
        check_ids = self.motor_ids + self.tool_ids

        for dev_id in check_ids:
            if self.bus.ping(dev_id):
                active_ids.append(dev_id)
                if verbose:
                    logging.info(f"Dynamixel ID {dev_id} is active")
            else:
                if verbose and dev_id < self.DOF:
                    logging.warning(f"Dynamixel ID {dev_id} is NOT active")
        
        return active_ids

    def monitor_health(self, duration, interval=0.5):
        """
        Periodically pings all active motors for a given duration.
        Returns (True, None) if all OK, or (False, failed_id) on first failure.
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            for mid in self.active_ids:
                if not self.bus.ping(mid):
                    return False, mid
            time.sleep(interval)
        return True, None

    def set_target_position(self, q_target):
        if len(q_target) != self.DOF:
            logging.error(f"Target position length mismatch: expected {self.DOF}, got {len(q_target)}")
            return False
        
        def task():
            # 1. Disable Torque
            self.bus.group_sync_write_torque_enable(self.motor_ids, 0)
            # 2. Set Operating Mode
            self.bus.group_sync_write_operating_mode([(i, rby.DynamixelBus.CurrentBasedPositionControlMode) for i in self.motor_ids])
            # 3. Enable Torque
            self.bus.group_sync_write_torque_enable(self.motor_ids, 1)
            # 4. Send Position
            # Position scale: 4096 / (2*pi)
            raw_pos = [(i, int(q * 4096.0 / (2.0 * np.pi))) for i, q in enumerate(q_target)]
            self.bus.group_sync_write_send_position(raw_pos)

        if self.is_running:
            self.ev.push_task(task)
        else:
            task()
        return True
    def start_control(self, callback, safety_function=None):
        if not self.initialized:
            return False
        if self.ctrl_session_active:
            return False

        self.control_callback = callback
        self.safety_function = safety_function
        self.ctrl_session_active = True
        self.ctrl_callback_busy = False

        # Load robot kinematics
        self._init_dynamics()

        # Initial operating mode setup
        if not self.operating_mode_init:
            self.bus.group_sync_write_torque_enable(self.motor_ids, 0)
            self.bus.group_sync_write_operating_mode([(i, rby.DynamixelBus.CurrentControlMode) for i in self.motor_ids])
            self.bus.group_sync_write_torque_enable(self.motor_ids, 1)
            self.operating_mode_init = True

        self.ev.unpause()
        self.ctrl_ev.unpause()
        self.ev.start()
        self.ctrl_ev.start()

        self.ev.push_cyclic_task(self._ev_task, self.control_period)
        return True

    def stop_control(self, torque_disable=False):
        if not self.ctrl_session_active:
            return False

        # 1. Disable torque first if requested (Highest priority)
        if torque_disable:
            self.DisableTorque()

        # 2. Shutdown threads
        self.ctrl_session_active = False
        self.ev.stop()
        self.ctrl_ev.stop()

        self.control_callback = None
        return True

    def EnableTorque(self):
        self.bus.group_sync_write_torque_enable(self.motor_ids, 1)

    def DisableTorque(self):
        self.bus.group_sync_write_torque_enable(self.motor_ids, 0)

    def _ev_task(self):
        try:
            # 0. Reset faults for the current cycle
            self.state.fault_ids = []
            self.state.tool_fault_ids = []
            self.state.tool_warning_ids = []
            
            # 1. Read Tool buttons (Auxiliary - Non-fatal until threshold)
            for tid in self.active_tool_ids:
                res = self.bus.read_button_status(tid)
                if res:
                    _, bstate = res
                    if tid == self.RIGHT_TOOL_ID:
                        self.state.button_right = bstate
                    else:
                        self.state.button_left = bstate
                    self.tool_error_counts[tid] = 0 # Reset count on success
                else:
                    self.state.tool_warning_ids.append(tid) # Immediate warning for any slip
                    logging.warning(f"Tool ID {tid} skipped communication step (Count: {self.tool_error_counts[tid]+1})")
                    self.tool_error_counts[tid] += 1
                    if self.tool_error_counts[tid] >= self.MAX_TOOL_RETRIES:
                        self.state.tool_fault_ids.append(tid)

            # 2. Read Operating Modes (Joints Only - Critical)
            if self.bus_flag and self.active_joint_ids:
                temp_modes = self.bus.group_fast_sync_read_operating_mode(self.active_joint_ids, True)
                if temp_modes is not None:
                    if len(temp_modes) != len(self.active_joint_ids):
                        responded_ids = {mid for mid, _ in temp_modes}
                        self.state.fault_ids = [mid for mid in self.active_joint_ids if mid not in responded_ids]
                    else:
                        for mid, mode in temp_modes:
                            if mid < self.DOF:
                                self.state.operating_mode[mid] = mode
                else:
                    self.state.fault_ids = list(self.active_joint_ids)

            # 3. Read Motor States (Joints Only - Critical)
            if not self.state.fault_ids:
                ms_list = self.bus.get_motor_states(self.motor_ids)
                if ms_list:
                    if len(ms_list) != len(self.motor_ids):
                        responded_ids = {mid for mid, _ in ms_list}
                        self.state.fault_ids = [mid for mid in self.motor_ids if mid not in responded_ids]
                    else:
                        for mid, mstate in ms_list:
                            if mid < self.DOF:
                                self.state.q_joint[mid] = mstate.position
                                self.state.qvel_joint[mid] = mstate.velocity
                                self.state.current[mid] = mstate.current
                                self.state.torque_joint[mid] = mstate.current * self.torque_constant[mid]
                                if self.temp_flag:
                                    self.state.temperatures[mid] = mstate.temperature
                                else:
                                    self.state.temperatures[mid] = 0.0
                else:
                    self.state.fault_ids = list(self.motor_ids)

            # 4. Read Goal Positions - Removed to prevent overwriting user commands
            # if self.bus_flag and not self.state.fault_ids:
            #     temp_gp = self.bus.group_fast_sync_read(self.motor_ids, rby.DynamixelBus.AddrGoalPosition, 4)
            #     if temp_gp:
            #         for mid, val in temp_gp:
            #             if mid < self.DOF:
            #                 self.state.target_position[mid] = val / 4096.0 * 2.0 * np.pi
            #     else:
            #         self.state.fault_ids = list(self.motor_ids)

            # 5. Compute Kinematics & Dynamics
            if not self.state.fault_ids:
                # 5-1. Validate Input Data
                if not np.all(np.isfinite(self.state.q_joint)):
                    logging.error(f"[LeaderArm] ERROR: Non-finite joint data detected: {self.state.q_joint}")
                    self.state.fault_ids = list(range(self.DOF))
                else:
                    # 5-2. Re-ordering (L/R Swap Fix)
                    # Natural Order: 0..6 (Right), 7..13 (Left)
                    # URDF Order: J7..J13 (Left), J0..J6 (Right)
                    q_urdf = np.concatenate([self.state.q_joint[7:], self.state.q_joint[:7]])
                    
                    self.dyn_state.set_q(q_urdf)
                    self.robot.compute_forward_kinematics(self.dyn_state)
                    
                    # 5-3. Compute Gravity & Un-map back
                    grav_urdf = self.robot.compute_gravity_term(self.dyn_state)
                    # Un-map to Natural Order [Right, Left]
                    self.state.gravity_term = np.concatenate([grav_urdf[7:], grav_urdf[:7]]) * self.TORQUE_SCALING
                    
                    self.state.T_right = self.robot.compute_transformation(self.dyn_state, self.kBaseLinkId, self.kRightLinkId)
                    self.state.T_left = self.robot.compute_transformation(self.dyn_state, self.kBaseLinkId, self.kLeftLinkId)

            # 6. Safety Check & Control
            self.state.tool_error_counts = dict(self.tool_error_counts)

            # Treat both joint faults and tool faults as critical safety events
            if self.state.fault_ids or self.state.tool_fault_ids:
                if self.safety_function:
                    # Trigger user-defined safety behavior (e.g., Power Off)
                    self.safety_function(self.state)
                else:
                    # Fallback: Print combined error and skip cycle
                    all_faults = self.state.fault_ids + self.state.tool_fault_ids
                    print(f"[LeaderArm] ERROR: Hardware fault detected (IDs: {all_faults}) but no safety_function is registered! Skipping cycle.")
                return

            if self.control_callback and self.ctrl_session_active and not self.ctrl_callback_busy:
                self.ctrl_callback_busy = True
                
                # Capturing a snapshot of the state ensures that the control task 
                # works with a consistent snapshot, avoiding race conditions 
                # when _ev_task starts updating the state for the next cycle.
                captured_state = self.state.copy()
                self.ctrl_ev.push_task(lambda: self._ctrl_task(captured_state))
        except Exception as e:
            logging.error(f"[LeaderArm] UNEXPECTED ENGINE EXCEPTION: {e}")
            if self.safety_function:
                # Force safety shutdown on any software exception
                self.safety_function(self.state)
            raise e

    def _ctrl_task(self, state):
        try:
            user_input = self.control_callback(state)
            if user_input:
                self._handle_control_input(user_input, state)
        except Exception as e:
            logging.error(f"[LeaderArm] EXCEPTION IN CONTROL CALLBACK: {e}")
            if self.safety_function:
                self.safety_function(state)
            raise e
        finally:
            self.ctrl_callback_busy = False

    def _handle_control_input(self, user_input, state):

        changed_ids = []
        changed_id_modes = []
        id_position = []
        id_torque = []

        for i in range(self.DOF):
            if state.operating_mode[i] != user_input.target_operating_mode[i]:
                changed_ids.append(i)
                changed_id_modes.append((i, user_input.target_operating_mode[i]))
            else:
                if state.operating_mode[i] == rby.DynamixelBus.CurrentControlMode:
                    id_torque.append((i, user_input.target_torque[i]))
                elif state.operating_mode[i] == rby.DynamixelBus.CurrentBasedPositionControlMode:
                    id_torque.append((i, user_input.target_torque[i]))
                    id_position.append((i, user_input.target_position[i]))

        # Push write task back to ev thread
        self.ev.push_task(lambda: self._write_task(changed_ids, changed_id_modes, id_torque, id_position))
        self.ctrl_running_flag = False

    def _write_task(self, changed_ids, changed_id_modes, id_torque, id_position):
        try:
            if changed_ids:
                self.bus.group_sync_write_torque_enable(changed_ids, 0)
                self.bus.group_sync_write_operating_mode(changed_id_modes)
                self.bus.group_sync_write_torque_enable(changed_ids, 1)

            if id_torque:
                self.bus.group_sync_write_send_torque(id_torque)
            if id_position:
                self.bus.group_sync_write_send_position(id_position)
        except Exception as e:
            logging.error(f"[LeaderArm] EXCEPTION IN WRITE TASK: {e}")
            if self.safety_function:
                self.safety_function(self.state)
            raise e

    def close(self):
        self.stop_control(torque_disable=True)

class Gripper:
    """
    Class for gripper
    """

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
                print(f"Temperature_right: {self.bus.read_temperature(0)}")
                print(f"Temperature_left: {self.bus.read_temperature(1)}")
            time.sleep(0.1)

    def set_target(self, normalized_q):
        # self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        if not np.isfinite(self.min_q).all() or not np.isfinite(self.max_q).all():
            logging.error("Cannot set target. min_q or max_q is not valid.")
            return

        if GRIPPER_DIRECTION:
            self.target_q = normalized_q * (self.max_q - self.min_q) + self.min_q
        else:
            self.target_q = (1 - normalized_q) * (self.max_q - self.min_q) + self.min_q
            

