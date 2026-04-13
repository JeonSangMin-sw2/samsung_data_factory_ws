import os
import rby1_sdk as rby
import numpy as np
import argparse
import time
import datetime
import signal
import threading

URDF_PATH = "/models/master_arm/model.urdf"
LEADER_ARM_DEVICE_NAME = "/dev/rby1_master_arm"

class LeaderArm:
    DOF = 14
    DEVICE_COUNT = 16
    RIGHT_TOOL_ID = 0x80
    LEFT_TOOL_ID = 0x81
    TORQUE_SCALING = 0.46

    initialized_ = False
    ctrl_running_ = False
    is_running_ = False
    state_updated_ = False
    operating_mode_init_ = False

    ev_ = rby.base.EventLoop()
    ctrl_ev_ = rby.base.EventLoop()
    torque_constant_ = {1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043,
                        1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043} # Default torque constant

    dyn_robot_ = rby.dynamics.LoadRobotFromURDF(URDF_PATH, "Base")
    dyn_state_ = dyn_robot_.make_state(
        ["Base", "Link_0R", "Link_1R", "Link_2R", "Link_3R", "Link_4R", "Link_5R", "Link_6R", "Link_0L", "Link_1L",
         "Link_2L", "Link_3L", "Link_4L", "Link_5L", "Link_6L"],
        ["J0_Shoulder_Pitch_R", "J1_Shoulder_Roll_R", "J2_Shoulder_Yaw_R", "J3_Elbow_R", "J4_Wrist_Yaw1_R",
         "J5_Wrist_Pitch_R", "J6_Wrist_Yaw2_R", "J7_Shoulder_Pitch_L", "J8_Shoulder_Roll_L", "J9_Shoulder_Yaw_L",
         "J10_Elbow_L", "J11_Wrist_Yaw1_L", "J12_Wrist_Pitch_L", "J13_Wrist_Yaw2_L"]
    )
    dyn_state_.set_gravity([0, 0, -9.81])

    
    active_ids_ = []
    
    state_ = State()
    

    control_ = ControlInput(self, state_)
    
    
    class State:
        q_joint = np.zeros(DOF)
        qvel_joint = np.zeros(DOF)
        torque_joint = np.zeros(DOF)
        gravity_term = np.zeros(DOF)
        temperatures = np.zeros(DOF)
        button_right = rby.DynamixelBus.ButtonState()
        button_left = rby.DynamixelBus.ButtonState()
        T_right = np.eye(4)
        T_left = np.eye(4)

    class ControlInput:
        target_torque = np.zeros(DOF)
        target_position = np.zeros(DOF)
        target_torque = np.zeros(DOF)
        
    def __init__(self, dev_name):
        self.bus = rby.DynamixelBus(dev_name)
        self.control_period = 0.01
        self.torque_constant = np.arrays([1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043,
                                         1.6591, 1.6591, 1.6591, 1.3043, 1.3043, 1.3043, 1.3043])
        self.active_ids = []
        self.motor_ids = []
        self.tool_ids = [self.RIGHT_TOOL_ID, self.LEFT_TOOL_ID]
        
        self.robot = None
        self.dyn_state = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # State
        self.q_joint = np.zeros(self.DOF)
        self.qvel_joint = np.zeros(self.DOF)
        self.torque_joint = np.zeros(self.DOF)
        self.gravity_term = np.zeros(self.DOF)
        self.temperatures = {}
        self.button_right = rby.DynamixelBus.ButtonState()
        self.button_left = rby.DynamixelBus.ButtonState()
    
    def SetControlPeriod(self, control_period):
        self.control_period = control_period

    def SetModelPath(self, model_path):
        self.robot = rby.dynamics.LoadRobotFromURDF(model_path, "Base")
        self.dyn_state = self.robot.make_state(
            ["Base", "Link_0R", "Link_1R", "Link_2R", "Link_3R", "Link_4R", "Link_5R", "Link_6R", "Link_0L", "Link_1L",
             "Link_2L", "Link_3L", "Link_4L", "Link_5L", "Link_6L"],
            ["J0_Shoulder_Pitch_R", "J1_Shoulder_Roll_R", "J2_Shoulder_Yaw_R", "J3_Elbow_R", "J4_Wrist_Yaw1_R",
             "J5_Wrist_Pitch_R", "J6_Wrist_Yaw2_R", "J7_Shoulder_Pitch_L", "J8_Shoulder_Roll_L", "J9_Shoulder_Yaw_L",
             "J10_Elbow_L", "J11_Wrist_Yaw1_L", "J12_Wrist_Pitch_L", "J13_Wrist_Yaw2_L"]
        )
        self.dyn_state.set_gravity([0, 0, -9.81])

    def SetTorqueConstant(self, torque_constant):
        self.torque_constant = torque_constant

    def initialize(self, verbose=False):
        if not self.bus.open_port():
            print("Failed to open port")
            return []
        if not self.bus.set_baud_rate(self.bus.kDefaultBaudrate):
            print("Failed to set baud rate")
            return []
        
        self.active_ids = []
        for i in range(self.DOF):
            if self.bus.ping(i):
                self.active_ids.append(i)
                self.motor_ids.append(i)
                if verbose: print(f"Motor ID {i} is active")
        
        for i in self.tool_ids:
            if self.bus.ping(i):
                self.active_ids.append(i)
                if verbose: print(f"Tool ID {i} is active")
        
        self.bus.set_torque_constant(self.torque_constant.tolist())
        return self.active_ids

    def start_control(self, callback):
        self.is_running = True
        self.callback = callback
        self.thread = threading.Thread(target=self._loop)
        self.thread.start()

    def stop_control(self):
        self.is_running = False
        if self.thread:
            self.thread.join()

    def EnableTorque(self):
        self.bus.group_sync_write_torque_enable(self.motor_ids, 1)

    def DisableTorque(self):
        self.bus.group_sync_write_torque_enable(self.motor_ids, 0)

    def _loop(self):
        # Initial mode setup
        self.bus.group_sync_write_torque_enable(self.motor_ids, 0)
        self.bus.group_sync_write_operating_mode([(i, rby.DynamixelBus.CurrentControlMode) for i in self.motor_ids])
        self.bus.group_sync_write_torque_enable(self.motor_ids, 1)

        while self.is_running:
            start_time = time.time()
            
            # 1. Read Tool buttons
            for tid in self.tool_ids:
                if tid in self.active_ids:
                    res = self.bus.read_button_status(tid)
                    if res:
                        _, state = res
                        if tid == self.RIGHT_TOOL_ID: self.button_right = state
                        else: self.button_left = state

            # 2. Read Motor States
            ms_list = self.bus.get_motor_states(self.motor_ids)
            if ms_list:
                for mid, mstate in ms_list:
                    self.q_joint[mid] = mstate.position
                    self.qvel_joint[mid] = mstate.velocity
                    self.torque_joint[mid] = mstate.current * self.torque_constant[mid]
                    self.temperatures[mid] = mstate.temperature
            else:
                continue # Skip loop if communication fails

            # 3. Compute Gravity
            self.dyn_state.set_q(self.q_joint)
            self.robot.compute_forward_kinematics(self.dyn_state)
            self.gravity_term = self.robot.compute_gravity_term(self.dyn_state) * self.TORQUE_SCALING

            # 4. User Callback
            # Minimal state object for the callback
            class StateMock:
                pass
            s = StateMock()
            s.q_joint = self.q_joint
            s.gravity_term = self.gravity_term
            s.button_right = self.button_right
            s.button_left = self.button_left
            s.temperatures = self.temperatures

            user_input = self.callback(s)

            # 5. Write Commands
            # For simplicity in QC script, we only use current control
            self.bus.group_sync_write_send_torque(list(enumerate(user_input.target_torque.tolist())))

            elapsed = time.time() - start_time
            sleep_time = self.control_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self):
        self.stop_control()
        if initialized_:
            self.disable_torque()

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
            

