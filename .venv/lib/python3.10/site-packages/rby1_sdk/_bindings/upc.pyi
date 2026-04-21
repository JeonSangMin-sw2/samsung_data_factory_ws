"""

Module for controlling and communicating with devices.

Provides utilities for controlling master arm and other devices.
"""
from __future__ import annotations
import _bindings
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__: list[str] = ['GripperDeviceName', 'MasterArm', 'MasterArmDeviceName', 'initialize_device']
class MasterArm:
    """
    
    Master arm control interface.
    
    This class provides control interface for a master arm device
    with 14 degrees of freedom, including joint control, gravity compensation,
    and button/trigger input handling.
    
    Attributes
    ----------
    DOF : int
        Number of degrees of freedom (14).
    DeviceCount : int
        Total number of devices including tools (16).
    TorqueScaling : float
        Torque scaling factor for gravity compensation (0.5).
    MaximumTorque : float
        Maximum allowed torque in Nm (4.0).
    RightToolId : int
        Device ID for right tool (0x80).
    LeftToolId : int
        Device ID for left tool (0x81).
    """
    class ControlInput:
        """
        
        Master arm control input.
        
        This class represents the control input for the master arm
        including target operating modes, positions, and torques.
        
        Attributes
        ----------
        target_operating_mode : numpy.ndarray, shape (14,), dtype=int32
            Target operating modes for each joint.
        target_position : numpy.ndarray, shape (14,), dtype=float64
            Target positions for each joint [rad].
        target_torque : numpy.ndarray, shape (14,), dtype=float64
            Target torques for each joint [Nm].
        """
        target_operating_mode: numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.int32]]
        target_position: numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]
        target_torque: numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]
        def __init__(self) -> None:
            """
            Construct a ``ControlInput`` instance with default values.
            """
    class State:
        """
        
        Master arm state information.
        
        This class represents the current state of the master arm
        including joint positions, velocities, torques, and tool states.
        
        Attributes
        ----------
        q_joint : numpy.ndarray, shape (14,), dtype=float64
            Joint positions [rad].
        qvel_joint : numpy.ndarray, shape (14,), dtype=float64
            Joint velocities [rad/s].
        torque_joint : numpy.ndarray, shape (14,), dtype=float64
            Joint torques [Nm].
        gravity_term : numpy.ndarray, shape (14,), dtype=float64
            Gravity compensation terms.
        operating_mode : numpy.ndarray, shape (14,), dtype=int32
            Operating modes for each joint.
        button_right : ButtonState
            Right tool button and trigger state.
        button_left : ButtonState
            Left tool button and trigger state.
        T_right : numpy.ndarray, shape (4, 4), dtype=float64
            Right tool transformation matrix (SE(3)) with respect to the master arm base.
        T_left : numpy.ndarray, shape (4, 4), dtype=float64
            Left tool transformation matrix (SE(3)) with respect to the master arm base.
        """
        def __init__(self) -> None:
            """
            Construct a ``State`` instance with default values.
            """
        def __repr__(self) -> str:
            ...
        def __str__(self) -> str:
            ...
        @property
        def T_left(self) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def T_right(self) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def button_left(self) -> _bindings.DynamixelBus.ButtonState:
            ...
        @property
        def button_right(self) -> _bindings.DynamixelBus.ButtonState:
            ...
        @property
        def gravity_term(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def operating_mode(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.int32]]:
            ...
        @property
        def q_joint(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def qvel_joint(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def target_position(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]:
            ...
        @property
        def torque_joint(self) -> numpy.ndarray[tuple[typing.Literal[14], typing.Literal[1]], numpy.dtype[numpy.float64]]:
            ...
    DOF: typing.ClassVar[int] = 14
    DeviceCount: typing.ClassVar[int] = 16
    LeftToolId: typing.ClassVar[int] = 129
    MaximumTorque: typing.ClassVar[float] = 4.0
    RightToolId: typing.ClassVar[int] = 128
    TorqueScaling: typing.ClassVar[float] = 0.5
    def __init__(self, dev_name: str = '/dev/rby1_master_arm') -> None:
        """
        Construct a ``MasterArm`` instance.
        
        Parameters
        ----------
        dev_name : str, optional
            Device name. Default is ``/dev/rby1_master_arm``'.
        """
    def __repr__(self: MasterArm.ControlInput) -> str:
        ...
    def __str__(self: MasterArm.ControlInput) -> str:
        ...
    def disable_torque(self) -> bool:
        """
        Disable torque of motors
        """
    def enable_torque(self) -> bool:
        """
        Enable torque of motors
        """
    def initialize(self, verbose: bool = False) -> list[int]:
        """
        Initialize the master arm and detect active devices.
        
        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose output. Default is False.
        
        Returns
        -------
        list[int]
            List of active device IDs.
        """
    def set_control_period(self, control_period: float) -> None:
        """
        Set the control update period.
        
        Parameters
        ----------
        control_period : float
            Control period in seconds.
        """
    def set_model_path(self, model_path: str) -> None:
        """
        Set the path to the URDF model file.
        
        Parameters
        ----------
        model_path : str
            Path to the URDF model file.
        """
    def set_torque_constant(self, torque_constant: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(14)]) -> None:
        """
        Set torque constant.
        
        Parameters
        ----------
        torque_constant : numpy.ndarray (14, )
        
        Parameters
        ----------
        model_path : str
            Path to the URDF model file.
        """
    def start_control(self, control: typing.Callable[[MasterArm.State], MasterArm.ControlInput]) -> bool:
        """
        Start the control loop.
        
        Parameters
        ----------
        control : callable, optional
            Control callback function that takes State and returns ControlInput.
            If None, no control is applied.
        
        Returns
        -------
        bool
        
        Examples
        --------
        >>> master_arm = rby.upc.MasterArm(rby.upc.MasterArmDeviceName)
        >>> master_arm.set_model_path("model.urdf") # path/to/master_arm_model.urdf
        >>> master_arm.set_control_period(0.01)
        >>> active_ids = master_arm.initialize(verbose=True)
        >>> if len(active_ids) != rby.upc.MasterArm.DeviceCount:
        ...     print("Error: Mismatch in the number of devices detected for RBY Master Arm.")
        ...     exit(1)
        >>>
        >>> def control(state: rby.upc.MasterArm.State):
        ...     with np.printoptions(suppress=True, precision=3, linewidth=300):
        ...         print(f"--- {datetime.datetime.now().time()} ---")
        ...         print(f"q: {state.q_joint}")
        ...         print(f"g: {state.gravity_term}")
        ...         print(
        ...             f"right: {state.button_right.button}, left: {state.button_left.button}"
        ...         )
        ...     input = rby.upc.MasterArm.ControlInput()
        ...     input.target_operating_mode.fill(rby.DynamixelBus.CurrentControlMode)
        ...     input.target_torque = state.gravity_term
        ...     return input
        >>>
        >>> master_arm.start_control(control)
        >>> time.sleep(100)
        """
    def stop_control(self, torque_disable: bool = False) -> bool:
        """
        Stop the control loop.
        
        Parameters
        ----------
        torque_disable : bool, optional
        
        Returns
        -------
        bool
        """
def initialize_device(device_name: str) -> None:
    """
    initialize_device(device_name)
    
    Initialize a device with the given name.
    
    Sets the latency timer of the device to 1.
    
    Args:
        device_name (str): Name of the device to initialize (e.g., '/dev/ttyUSB0', '/dev/rby1_master_arm').
    
    Returns:
        bool: True if device initialized successfully, False otherwise.
    """
GripperDeviceName: str = '/dev/rby1_gripper'
MasterArmDeviceName: str = '/dev/rby1_master_arm'
