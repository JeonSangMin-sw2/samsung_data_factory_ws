"""

Robot dynamics and kinematics module.

Provides robot dynamics calculations including inertia,
joint dynamics, and link properties.
"""
from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import typing
__all__: list[str] = ['Collision', 'CollisionResult', 'Geom', 'GeomCapsule', 'GeomType', 'Joint', 'Link', 'MobileBase', 'MobileBaseDifferential', 'MobileBaseType', 'Robot', 'RobotConfiguration', 'Robot_18', 'Robot_24', 'Robot_26', 'State', 'State_18', 'State_24', 'State_26', 'load_robot_from_urdf', 'load_robot_from_urdf_data']
M = typing.TypeVar("M", bound=int)
N = typing.TypeVar("N", bound=int)
class Collision:
    """
    
    Collision detection object.
    
    Manages collision detection for a link with multiple geometric objects.
    
    Attributes
    ----------
    origin : numpy.ndarray
        Origin transformation matrix.
    geoms : list
        List of geometric objects for collision detection.
    """
    def __init__(self, name: str) -> None:
        """
        Construct a ``Collision`` instance.
        
        Parameters
        ----------
        name : str
            Name of the collision object.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def add_geom(self, geom: Geom) -> None:
        """
        Add a geometric object for collision detection.
        
        Parameters
        ----------
        geom : Geom
            Geometric object to add.
        """
    @typing.overload
    def get_geoms(self) -> list[Geom]:
        """
        Get the list of geometric objects.
        
        Returns
        -------
        list
            List of geometric objects.
        """
    @typing.overload
    def get_geoms(self) -> list[Geom]:
        """
        Get the list of geometric objects (const version).
        
        Returns
        -------
        list
            List of geometric objects.
        """
    def get_origin(self) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        """
        Get the origin transformation.
        
        Returns
        -------
        numpy.ndarray
            Origin transformation matrix.
        """
    def set_origin(self, T: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the origin transformation.
        
        Parameters
        ----------
        T : numpy.ndarray
            Origin transformation matrix.
        """
class CollisionResult:
    """
    
    Collision detection result.
    
    Provides information about collision detection between two geometric objects.
    
    Attributes
    ----------
    link1 : str
        Name of the first link involved in collision.
    link2 : str
        Name of the second link involved in collision.
    position1 : numpy.ndarray
        Position of collision point on first link [m].
    position2 : numpy.ndarray
        Position of collision point on second link [m].
    distance : float
        Signed distance [m]. Positive when separated, ``0`` when touching,
        negative when overlapping (penetration depth = ``-distance``).
      
    """
    def __init__(self) -> None:
        """
              Construct a ``CollisionResult`` instance.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def distance(self) -> float:
        ...
    @property
    def link1(self) -> str:
        ...
    @property
    def link2(self) -> str:
        ...
    @property
    def position1(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @property
    def position2(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
class Geom:
    """
    
    Base geometry class.
    
    Abstract base class for geometric objects used in collision detection.
    
    Attributes
    ----------
    coltype : int
        Collision type identifier.
    colaffinity : int
        Collision affinity identifier.
      
    """
    def __init__(self, arg0: int, arg1: int) -> None:
        """
        Construct a ``Geom`` instance.
        
        Parameters
        ----------
        coltype : int
            Collision type identifier.
        colaffinity : int
            Collision affinity identifier.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compute_minimum_distance(self, T: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]], other_geom: Geom, other_T: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]) -> CollisionResult | None:
        """
        Compute the minimum Euclidean distance between this geometry and another geometry.
        
        Parameters
        ----------
        T : numpy.ndarray, shape (4, 4)
            Homogeneous transformation of this geometry in the world (SE(3)).
        other_geom : Geom
            The other geometry.
        other_T : numpy.ndarray, shape (4, 4)
            Homogeneous transformation of the other geometry in the world (SE(3)).
        
        Returns
        -------
        CollisionResult or None
            Signed-distance result with the two closest points **on each geometry**.
            Returns ``None`` if the pair is unsupported or filtered out.
        
        Notes
        -----
        The returned distance is signed:
        
        - distance > 0 : geometries are separated by that metric distance
        - distance = 0 : geometries are just touching
        - distance < 0 : geometries overlap; penetration depth = -distance
        
        Examples
        --------
        >>> # quick, copy-paste friendly
        >>> import numpy as np
        >>> import rby1_sdk.dynamics as dyn
        >>> 
        >>> caps1 = dyn.GeomCapsule(height=0.4, radius=0.05, coltype=0, colaffinity=0)
        >>> caps2 = dyn.GeomCapsule(height=0.4, radius=0.05, coltype=0, colaffinity=0)
        >>> T1 = np.eye(4)
        >>> T2 = np.eye(4); T2[0, 3] = 0.20  # shift 20 cm in x
        >>> 
        >>> res = caps1.compute_minimum_distance(T1, caps2, T2)
        >>> if res is not None:
        ...     print(res.distance)
        """
    def filter(self, other_geom: Geom) -> bool:
        """
        Filter collision detection with another geometry.
        
        Parameters
        ----------
        other_geom : Geom
            Other geometry to filter with.
        
        Returns
        -------
        bool
            True if collision should be checked, False otherwise.
        """
    def get_colaffinity(self) -> int:
        """
        Get the collision affinity.
        
        Returns
        -------
        int
            Collision affinity identifier.
        """
    def get_coltype(self) -> int:
        """
        Get the collision type.
        
        Returns
        -------
        int
            Collision type identifier.
        """
    def get_type(self) -> GeomType:
        """
        Get the geometry type.
        
        Returns
        -------
        GeomType
            Type of the geometry.
        """
class GeomCapsule(Geom):
    """
    
    Capsule geometry.
    
    Represents a capsule (cylinder with rounded ends) for collision detection.
    
    Attributes
    ----------
    start_point : numpy.ndarray
        Start point of the capsule axis [m].
    end_point : numpy.ndarray
        End point of the capsule axis [m].
    radius : float
        Radius of the capsule [m].
    """
    @typing.overload
    def __init__(self, height: float, radius: float, coltype: int, colaffinity: int) -> None:
        """
        Construct a ``GeomCapsule`` with height and radius.
        
        Parameters
        ----------
        height : float
            Height of the capsule [m].
        radius : float
            Radius of the capsule [m].
        coltype : int
            Collision type identifier.
        colaffinity : int
            Collision affinity identifier.
        """
    @typing.overload
    def __init__(self, start_point: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]], end_point: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]], radius: float, coltype: int, colaffinity: int) -> None:
        """
        Construct a ``GeomCapsule`` with start and end points.
        
        Parameters
        ----------
        start_point : numpy.ndarray
            Start point of the capsule axis [m].
        end_point : numpy.ndarray
            End point of the capsule axis [m].
        radius : float
            Radius of the capsule [m].
        coltype : int
            Collision type identifier.
        colaffinity : int
            Collision affinity identifier.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_end_point(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the end point of the capsule axis.
        
        Returns
        -------
        numpy.ndarray
            End point with respect to link frame.
        """
    def get_radius(self) -> float:
        """
        Get the radius of the capsule.
        
        Returns
        -------
        float
            Radius [m].
        """
    def get_start_point(self) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the start point of the capsule axis.
        
        Returns
        -------
        numpy.ndarray
            Start point with respect to link frame.
        """
class GeomType:
    """
    
    Geometry type enumeration.
    
    Defines the types of geometric objects supported.
    
    Members
    -------
    Capsule : int
        Capsule geometry type.
      
    
    Members:
    
      Capsule : 
          Capsule geometry type.
    """
    Capsule: typing.ClassVar[GeomType]  # value = <GeomType.Capsule: 0>
    __members__: typing.ClassVar[dict[str, GeomType]]  # value = {'Capsule': <GeomType.Capsule: 0>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Joint:
    """
    
    Joint in the robot dynamics model.
    
    Represents a joint connecting two links in the robot.
    
    Attributes
    ----------
    name : str
        Name of the joint.
    parent_link : Link
        Parent link of the joint.
    child_link : Link
        Child link of the joint.
    """
    @staticmethod
    def make(name: str, S: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> Joint:
        """
        Create a joint with a specific screw axis.
        
        Parameters
        ----------
        name : str
            Name of the joint.
        S : numpy.ndarray
            6D screw axis vector.
        """
    @staticmethod
    def make_fixed(name: str) -> Joint:
        """
        Create a fixed joint.
        
        Parameters
        ----------
        name : str
            Name of the joint.
        """
    @staticmethod
    def make_prismatic(name: str, T: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]] = ..., axis: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]] = ...) -> Joint:
        """
        Create a prismatic joint.
        
        Parameters
        ----------
        name : str
            Name of the joint.
        T : numpy.ndarray, optional
            Transformation from the parent link's frame to the joint's frame. Defaults to identity.
        axis : numpy.ndarray, optional
            Axis of translation. Defaults to [0, 0, 1].
        """
    @staticmethod
    def make_revolute(name: str, T: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]] = ..., axis: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]] = ...) -> Joint:
        """
        Create a revolute joint.
        
        Parameters
        ----------
        name : str
            Name of the joint.
        T : numpy.ndarray, optional
            Transformation from the parent link's frame to the joint's frame. Defaults to identity.
        axis : numpy.ndarray, optional
            Axis of rotation. Defaults to [0, 0, 1].
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def connect_links(self, parent_link: Link, child_link: Link, T_pj: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]] = ..., T_jc: numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]] = ...) -> None:
        """
        Connect two links through this joint.
        
        Parameters
        ----------
        parent_link : Link
            Parent link.
        child_link : Link
            Child link.
        T_pj : numpy.ndarray, shape (4, 4), dtype=float64, optional
            Transformation from parent joint to joint. Default is identity.
        T_jc : numpy.ndarray, shape (4, 4), dtype=float64, optional
            Transformation from joint to child joint. Default is identity.
        """
    def disconnect(self) -> None:
        """
        Disconnect the joint from its parent and child links.
        """
    @typing.overload
    def get_child_link(self) -> Link:
        """
        Get the child link of the joint.
        
        Returns
        -------
        Link
            Child link.
        """
    @typing.overload
    def get_child_link(self) -> Link:
        """
        Get the child link of the joint (const version).
        
        Returns
        -------
        Link
            Child link.
        """
    def get_limit_q_lower(self) -> float:
        """
        Get lower joint position limit.
        
        Returns
        -------
        float
            Lower limit for joint position [rad].
        """
    def get_limit_q_upper(self) -> float:
        """
        Get upper joint position limit.
        
        Returns
        -------
        float
            Upper limit for joint position [rad].
        """
    def get_limit_qddot_lower(self) -> float:
        """
        Get lower joint acceleration limit.
        
        Returns
        -------
        float
            Lower limit for joint acceleration [rad/s²].
        """
    def get_limit_qddot_upper(self) -> float:
        """
        Get upper joint acceleration limit.
        
        Returns
        -------
        float
            Upper limit for joint acceleration [rad/s²].
        """
    def get_limit_qdot_lower(self) -> float:
        """
        Get lower joint velocity limit.
        
        Returns
        -------
        float
            Lower limit for joint velocity [rad/s].
        """
    def get_limit_qdot_upper(self) -> float:
        """
        Get upper joint velocity limit.
        
        Returns
        -------
        float
            Upper limit for joint velocity [rad/s].
        """
    def get_limit_torque(self) -> float:
        """
        Get joint torque limits.
        
        Returns
        -------
        float
            Torque limits [Nm].
        """
    def get_name(self) -> str:
        """
        Get the name of the joint.
        
        Returns
        -------
        str
            Name of the joint.
        """
    def get_parent_link(self) -> Link:
        """
        Get the parent link of the joint.
        
        Returns
        -------
        Link, optional
            Parent link if it exists, None otherwise.
        """
    def is_fixed(self) -> bool:
        """
        Check if the joint is fixed.
        
        Returns
        -------
        bool
            True if fixed, False otherwise.
        """
    def set_limit_q(self, lower: float, upper: float) -> None:
        """
        Set joint position limits.
        
        Parameters
        ----------
        lower : float
            Lower limit for joint position [rad].
        upper : float
            Upper limit for joint position [rad].
        """
    def set_limit_q_lower(self, val: float) -> None:
        """
        Set lower joint position limit.
        
        Parameters
        ----------
        val : float
            New lower limit for joint position [rad].
        """
    def set_limit_q_upper(self, val: float) -> None:
        """
        Set upper joint position limit.
        
        Parameters
        ----------
        val : float
            New upper limit for joint position [rad].
        """
    def set_limit_qddot(self, lower: float, upper: float) -> None:
        """
        Set joint acceleration limits.
        
        Parameters
        ----------
        lower : float
            Lower limit for joint acceleration [rad/s²].
        upper : float
            Upper limit for joint acceleration [rad/s²].
        """
    def set_limit_qddot_lower(self, val: float) -> None:
        """
        Set lower joint acceleration limit.
        
        Parameters
        ----------
        val : float
            New lower limit for joint acceleration [rad/s²].
        """
    def set_limit_qddot_upper(self, val: float) -> None:
        """
        Set upper joint acceleration limit.
        
        Parameters
        ----------
        val : float
            New upper limit for joint acceleration [rad/s²].
        """
    def set_limit_qdot(self, lower: float, upper: float) -> None:
        """
        Set joint velocity limits.
        
        Parameters
        ----------
        lower : float
            Lower limit for joint velocity [rad/s].
        upper : float
            Upper limit for joint velocity [rad/s].
        """
    def set_limit_qdot_lower(self, val: float) -> None:
        """
        Set lower joint velocity limit.
        
        Parameters
        ----------
        val : float
            New lower limit for joint velocity [rad/s].
        """
    def set_limit_qdot_upper(self, val: float) -> None:
        """
        Set upper joint velocity limit.
        
        Parameters
        ----------
        val : float
            New upper limit for joint velocity [rad/s].
        """
    def set_limit_torque(self, torque: float) -> None:
        """
        Set joint torque limits.
        
        Parameters
        ----------
        torque : float
            Torque limits.
        """
class Link:
    """
    
    Link in the robot dynamics model.
    
    Represents a rigid body link in the robot.
    
    Attributes
    ----------
    name : str
        Name of the link.
    inertial : Inertial
        Inertial properties of the link.
    parent_joint : Joint
        Parent joint of the link.
    """
    def __init__(self, name: str, I: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]] = ...) -> None:
        """
        Construct a ``Link`` instance.
        
        Parameters
        ----------
        name : str
            Name of the link.
        I : Inertial, optional
            Inertial properties. Default is identity.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def add_collision(self, collision: Collision) -> None:
        """
        Add a collision object to the link.
        
        Parameters
        ----------
        collision : Collision
            Collision object to add.
        """
    @typing.overload
    def get_child_joint_list(self) -> list[Joint]:
        """
        Get the list of child joints.
        
        Returns
        -------
        list
            List of child joints.
        """
    @typing.overload
    def get_child_joint_list(self) -> list[Joint]:
        """
        Get the list of child joints (const version).
        
        Returns
        -------
        list
            List of child joints.
        """
    @typing.overload
    def get_collisions(self) -> list[Collision]:
        """
        Get the list of collisions.
        
        Returns
        -------
        list
            List of collisions.
        """
    @typing.overload
    def get_collisions(self) -> list[Collision]:
        """
        Get the list of collisions (const version).
        
        Returns
        -------
        list
            List of collisions.
        """
    def get_name(self) -> str:
        """
        Get the name of the link.
        
        Returns
        -------
        str
            Name of the link.
        """
    def get_parent_joint(self) -> Joint:
        """
        Get the parent joint of the link.
        
        Returns
        -------
        Joint, optional
            Parent joint if it exists, None otherwise.
        """
class MobileBase:
    """
    
    Base class for mobile bases.
    
    Represents the base of the robot that can move.
    
    Attributes
    ----------
    type : MobileBaseType
        Type of the mobile base (e.g., differential, mecanum).
    T : numpy.ndarray
        Transformation matrix from the base to the world frame.
    joints : list
        List of joints that make up the mobile base.
    params : dict
        Parameters of the mobile base.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def T(self) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        ...
    @property
    def joints(self) -> list[str]:
        ...
    @property
    def params(self) -> list[float]:
        ...
    @property
    def type(self) -> MobileBaseType:
        ...
class MobileBaseDifferential(MobileBase):
    """
    
    Differential drive mobile base.
    
    Represents a differential drive mobile base with two wheels.
    
    Attributes
    ----------
    right_wheel_idx : int
        Index of the right wheel joint.
    left_wheel_idx : int
        Index of the left wheel joint.
    wheel_base : float
        Distance between the two wheels [m].
    wheel_radius : float
        Radius of the wheels [m].
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def left_wheel_idx(self) -> int:
        ...
    @property
    def right_wheel_idx(self) -> int:
        ...
    @property
    def wheel_base(self) -> float:
        ...
    @property
    def wheel_radius(self) -> float:
        ...
class MobileBaseType:
    """
    
    Mobile base type enumeration.
    
    Defines the types of mobile bases supported.
    
    Members
    -------
    None : int
        No mobile base.
    Differential : int
        Differential drive mobile base.
    Mecanum : int
        Mecanum drive mobile base.
    
    
    Members:
    
      Unspecified : 
    No mobile base.
    
    
      Differential : 
    Differential drive mobile base.
    
    
      Mecanum : 
    Mecanum drive mobile base.
    """
    Differential: typing.ClassVar[MobileBaseType]  # value = <MobileBaseType.Differential: 1>
    Mecanum: typing.ClassVar[MobileBaseType]  # value = <MobileBaseType.Mecanum: 2>
    Unspecified: typing.ClassVar[MobileBaseType]  # value = <MobileBaseType.Unspecified: 0>
    __members__: typing.ClassVar[dict[str, MobileBaseType]]  # value = {'Unspecified': <MobileBaseType.Unspecified: 0>, 'Differential': <MobileBaseType.Differential: 1>, 'Mecanum': <MobileBaseType.Mecanum: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class Robot:
    """
    Robot dynamics model.
    
    Represents the dynamics of a robot with a given number of degrees of freedom.
    
    Attributes
    ----------
    base : Link
        Base link of the robot.
    link_names : list
        List of names of all links.
    joint_names : list
        List of names of all joints.
    """
    @staticmethod
    def count_joints(base_link: Link, include_fixed: bool = False) -> int:
        """
        count_joints(base_link, include_fixed=False)
        
        Counts the number of joints in a kinematic chain starting from a base link.
        
        Parameters
        ----------
        base_link : rby1_sdk.dynamics.Link
            The starting link of the kinematic chain.
        include_fixed : bool, optional
            Whether to include fixed joints in the count. Default is False.
        
        Returns
        -------
        int
            The total number of joints.
        """
    def __init__(self, robot_configuration: RobotConfiguration) -> None:
        """
        Construct a Robot instance.
        
        Parameters
        ----------
        robot_configuration : RobotConfiguration
            Configuration of the robot.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compute_2nd_diff_forward_kinematics(self, state: State) -> None:
        """
        Computes the second-order differential forward kinematics for each joint.
        
        This method calculates the body acceleration for each joint frame based on the
        current joint accelerations (`qddot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed body accelerations.
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        """
    def compute_body_jacobian(self, state: State, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], N], numpy.dtype[numpy.float64]]:
        """
        Computes the body Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF body Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_body_velocity(self, state: State, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the relative body velocity (twist) of a target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x1 body velocity vector (twist).
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of a single target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State, ref_link: int, target_links: list[int]) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the combined center of mass of multiple target links with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_links : list[int]
            A list of indices of the target links.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the combined center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the total center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], N], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of a single target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], N], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of the entire robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix for the whole robot.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_diff_forward_kinematics(self, state: State) -> None:
        """
        Computes the differential forward kinematics for each joint.
        
        This method calculates the body velocity (twist) for each joint frame based on
        the current joint velocities (`qdot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint velocities. This object
            will be updated with the computed body velocities.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        """
    def compute_forward_kinematics(self, state: State) -> None:
        """
        Computes the forward kinematics for each joint.
        
        This method calculates the transformation matrix from the base to each joint frame
        based on the current joint positions (`q`) in the state. The results are cached
        within the `state` object. This function must be called before other kinematics
        or dynamics calculations.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint positions. This object
            will be updated with the computed transformation matrices.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> # Assume dyn_robot is an initialized rby_dyn.Robot instance
        >>> # and dyn_state is a corresponding state object.
        >>> dyn_state.set_q(np.random.rand(dyn_robot.get_dof()))
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Now you can compute transformations, Jacobians, etc.
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)
        >>> print(transform)
        """
    def compute_gravity_term(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the gravity compensation term for the robot.
        
        This method calculates the joint torques required to counteract gravity at the
        current joint positions. The gravity vector must be set in the state object
        prior to calling this function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions and the gravity vector.
        
        Returns
        -------
        numpy.ndarray
            A vector of joint torques required to compensate for gravity.
        
        Notes
        -----
        - `compute_forward_kinematics` must be called before this function.
        - The gravity vector (spatial acceleration) must be set on the `state` object
          using `state.set_gravity()` or `state.set_Vdot0()`. For standard gravity along
          the negative Z-axis, the vector is `[0, 0, 0, 0, 0, -9.81]`.
        
        Examples
        --------
        >>> # Continuing from a previous example where dyn_robot and dyn_state are set up.
        >>> dyn_state.set_gravity(np.array([0, 0, 0, 0, 0, -9.81]))
        >>> # or dyn_state.set_Vdot0(np.array([0, 0, 0, 0, 0, 9.81]))  # Note that direction is reversed
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> gravity_torques = dyn_robot.compute_gravity_term(dyn_state)
        >>> print(gravity_torques)
        """
    def compute_inverse_dynamics(self, state: State) -> None:
        """
        Computes the inverse dynamics of the robot.
        
        This method calculates the joint torques required to achieve the given joint
        accelerations (`qddot`), considering the current joint positions (`q`) and
        velocities (`qdot`). The results are stored back into the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed joint torques.
            Hello. This is state.
        
        Notes
        -----
        `compute_forward_kinematics`, `compute_diff_forward_kinematics`, and
        `compute_2nd_diff_forward_kinematics` must be called in order before this function.
        
        Examples
        --------
        >>> # This example demonstrates the full sequence for inverse dynamics.
        >>> import rby1_sdk as rby
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> robot = rby.create_robot_a("localhost:50051")
        >>> robot.connect()
        >>> dyn_robot = robot.get_dynamics()
        >>> dyn_state = dyn_robot.make_state(
        ...     dyn_robot.get_link_names(), dyn_robot.get_joint_names()
        ... )
        >>> q = (np.random.rand(dyn_robot.get_dof()) - 0.5) * np.pi / 2
        >>> dyn_state.set_q(q)
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>>
        >>> # Perform kinematics calculations in order
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        >>>
        >>> # Compute inverse dynamics
        >>> dyn_robot.compute_inverse_dynamics(dyn_state)
        >>>
        >>> # Get the resulting torques
        >>> torques = dyn_state.get_tau()
        >>> with np.printoptions(precision=4, suppress=True):
        ...     print(f"Inverse dynamics torque (Nm): {torques}")
        """
    def compute_mass(self, state: State, target_link_index: int) -> float:
        """
        Computes the mass of a specific link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        float
            The mass of the specified link.
        """
    def compute_mass_matrix(self, state: State) -> numpy.ndarray[tuple[M, N], numpy.dtype[numpy.float64]]:
        """
        Computes the joint space mass matrix (inertia matrix) of the robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions.
        
        Returns
        -------
        numpy.ndarray
            The mass matrix (a square matrix of size DOF x DOF).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_mobility_diff_kinematics(self, state: State) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the forward differential kinematics for the mobile base.
        
        Calculates the linear and angular velocity of the mobile base from the current
        wheel velocities (`qdot`).
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including wheel velocities.
        
        Returns
        -------
        numpy.ndarray
            The resulting body velocity vector [w, vx, vy].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State, linear_velocity: numpy.ndarray[tuple[typing.Literal[2], typing.Literal[1]], numpy.dtype[numpy.float64]], angular_velocity: float) -> None:
        """
        Computes the inverse differential kinematics for the mobile base.
        
        Calculates the required wheel velocities to achieve a desired linear and angular
        velocity of the mobile base. Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        linear_velocity : numpy.ndarray
            The desired linear velocity (x, y) [m/s].
        angular_velocity : float
            The desired angular velocity (yaw) [rad/s].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State, body_velocity: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Computes the inverse differential kinematics for the mobile base from a body velocity vector.
        
        Calculates the required wheel velocities from a desired body velocity (twist).
        Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        body_velocity : numpy.ndarray
            The desired body velocity vector [w, vx, vy].
        """
    def compute_reflective_inertia(self, state: State, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the reflective inertia (task space inertia) of the target link with respect to the reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 reflective inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_space_jacobian(self, state: State, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], N], numpy.dtype[numpy.float64]]:
        """
        Computes the space Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF space Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_total_inertial(self, state: State, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the total spatial inertia of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 total spatial inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_transformation(self, state: State, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        """
        Computes the transformation matrix from a reference link to a target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link (the 'from' frame).
        target_link_index : int
            The index of the target link (the 'to' frame).
        
        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix (SE(3)).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def detect_collisions_or_nearest_links(self, state: State, collision_threshold: int = 0) -> list[CollisionResult]:
        """
        Detects collisions or finds the nearest links in the robot model.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        collision_threshold : int, optional
            The minimum number of link pairs to return. The function first finds all
            colliding pairs. If the number of colliding pairs is less than this
            threshold, it will supplement the result with the nearest non-colliding
            link pairs until the total count reaches the threshold. The returned list
            is always sorted by distance. If set to 0, only actual collisions are
            returned. Default is 0.
        
        Returns
        -------
        list[rby1_sdk.dynamics.CollisionResult]
            A list of collision results, sorted by distance.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def get_base(self) -> Link:
        """
        Get the base link of the robot.
        
        Returns
        -------
        Link
            Base link.
        """
    def get_dof(self) -> int:
        """
        Get the number of degrees of freedom.
        
        Returns
        -------
        int
            Number of degrees of freedom.
        """
    def get_joint_names(self) -> list[str]:
        """
        Get the list of names of all joints.
        
        Returns
        -------
        list
            List of joint names.
        """
    def get_joint_property(self, state: State, getter: typing.Callable[[Joint], float]) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets a specific property for all joints using a provided getter function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        getter : callable
            A function that takes a joint object and returns a double value.
        
        Returns
        -------
        numpy.ndarray
            A vector containing the specified property for each joint.
        """
    def get_limit_q_lower(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower position limits (q) for each joint.
        """
    def get_limit_q_upper(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper position limits (q) for each joint.
        """
    def get_limit_qddot_lower(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qddot_upper(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qdot_lower(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower velocity limits (q_dot) for each joint.
        """
    def get_limit_qdot_upper(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper velocity limits (q_dot) for each joint.
        """
    def get_limit_torque(self, state: State) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the torque limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of torque limits for each joint.
        """
    @typing.overload
    def get_link(self, name: str) -> Link:
        """
        Get a link by name.
        
        Parameters
        ----------
        name : str
            Name of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    @typing.overload
    def get_link(self, state: State, index: int) -> Link:
        """
        Get a link by state and index.
        
        Parameters
        ----------
        state : State
            Current state of the robot.
        index : int
            Index of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of names of all links.
        
        Returns
        -------
        list
            List of link names.
        """
    def get_number_of_joints(self) -> int:
        """
        Get the number of joints.
        
        Returns
        -------
        int
            Number of joints.
        """
    def make_state(self, link_names: list[str], joint_names: list[str]) -> State:
        """
        Create a state from link and joint names.
        
        The state object is essential for using the robot dynamics functions.
        It stores the robot's state, its state vector (e.g., indices of joints and links), 
        and also serves as a cache for intermediate results in dynamics and
        kinematics calculations to optimize for speed.
        
        Parameters
        ----------
        link_names : list[str]
            List of link names.
        joint_names : list[str]
            List of joint names.
        
        Returns
        -------
        State
            A new state object.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>>
        >>> link_0 = rby_dyn.Link("link_0")
        >>> link_1 = rby_dyn.Link("link_1")
        >>> 
        >>> joint_0 = rby_dyn.Joint.make_revolute("joint_0", np.identity(4), np.array([0, 0, 1]))
        >>> joint_0.connect_links(link_0, link_1, np.identity(4), np.identity(4))
        >>> 
        >>> dyn_robot = rby_dyn.Robot(
        ...     rby_dyn.RobotConfiguration(name="sample_robot", base_link=link_0)
        ... )
        >>> 
        >>> dyn_state = dyn_robot.make_state(["link_0", "link_1"], ["joint_0"])
        >>> dyn_state.set_q(np.array([np.pi / 2]))  # Angle of joint_0 is 90 degrees
        >>> 
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Calculate transformation from link_0 to link_1
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)  # 0: link_0, 1: link_1
        >>> print(transform)
        """
class RobotConfiguration:
    """
    
    Robot configuration.
    
    Defines the base link and mobile base of the robot.
    
    Attributes
    ----------
    name : str
        Name of the robot configuration.
    base_link : Link
        Base link of the robot.
    mobile_base : MobileBase
        Mobile base of the robot.
    """
    base_link: Link
    mobile_base: MobileBase
    name: str
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a ``RobotConfiguration`` instance.
        """
    @typing.overload
    def __init__(self, name: str, base_link: Link = None, mobile_base: MobileBase = None) -> None:
        """
        Construct a ``RobotConfiguration`` instance.
        
        Parameters
        ----------
        name : str
            Name of the robot configuration.
        base_link : Link
            Base link of the robot.
        mobile_base : MobileBase
            Mobile base of the robot.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class Robot_18:
    """
    Robot (DOF=18) dynamics model.
    
    Represents the dynamics of a robot with a given number of degrees of freedom.
    
    Attributes
    ----------
    base : Link
        Base link of the robot.
    link_names : list
        List of names of all links.
    joint_names : list
        List of names of all joints.
    """
    @staticmethod
    def count_joints(base_link: Link, include_fixed: bool = False) -> int:
        """
        count_joints(base_link, include_fixed=False)
        
        Counts the number of joints in a kinematic chain starting from a base link.
        
        Parameters
        ----------
        base_link : rby1_sdk.dynamics.Link
            The starting link of the kinematic chain.
        include_fixed : bool, optional
            Whether to include fixed joints in the count. Default is False.
        
        Returns
        -------
        int
            The total number of joints.
        """
    def __init__(self, robot_configuration: RobotConfiguration) -> None:
        """
        Construct a Robot instance.
        
        Parameters
        ----------
        robot_configuration : RobotConfiguration
            Configuration of the robot.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compute_2nd_diff_forward_kinematics(self, state: State_18) -> None:
        """
        Computes the second-order differential forward kinematics for each joint.
        
        This method calculates the body acceleration for each joint frame based on the
        current joint accelerations (`qddot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed body accelerations.
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        """
    def compute_body_jacobian(self, state: State_18, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[18]], numpy.dtype[numpy.float64]]:
        """
        Computes the body Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF body Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_body_velocity(self, state: State_18, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the relative body velocity (twist) of a target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x1 body velocity vector (twist).
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_18, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of a single target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_18, ref_link: int, target_links: list[int]) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the combined center of mass of multiple target links with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_links : list[int]
            A list of indices of the target links.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the combined center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_18, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the total center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_18, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[18]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of a single target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_18, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[18]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of the entire robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix for the whole robot.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_diff_forward_kinematics(self, state: State_18) -> None:
        """
        Computes the differential forward kinematics for each joint.
        
        This method calculates the body velocity (twist) for each joint frame based on
        the current joint velocities (`qdot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint velocities. This object
            will be updated with the computed body velocities.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        """
    def compute_forward_kinematics(self, state: State_18) -> None:
        """
        Computes the forward kinematics for each joint.
        
        This method calculates the transformation matrix from the base to each joint frame
        based on the current joint positions (`q`) in the state. The results are cached
        within the `state` object. This function must be called before other kinematics
        or dynamics calculations.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint positions. This object
            will be updated with the computed transformation matrices.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> # Assume dyn_robot is an initialized rby_dyn.Robot instance
        >>> # and dyn_state is a corresponding state object.
        >>> dyn_state.set_q(np.random.rand(dyn_robot.get_dof()))
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Now you can compute transformations, Jacobians, etc.
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)
        >>> print(transform)
        """
    def compute_gravity_term(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the gravity compensation term for the robot.
        
        This method calculates the joint torques required to counteract gravity at the
        current joint positions. The gravity vector must be set in the state object
        prior to calling this function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions and the gravity vector.
        
        Returns
        -------
        numpy.ndarray
            A vector of joint torques required to compensate for gravity.
        
        Notes
        -----
        - `compute_forward_kinematics` must be called before this function.
        - The gravity vector (spatial acceleration) must be set on the `state` object
          using `state.set_gravity()` or `state.set_Vdot0()`. For standard gravity along
          the negative Z-axis, the vector is `[0, 0, 0, 0, 0, -9.81]`.
        
        Examples
        --------
        >>> # Continuing from a previous example where dyn_robot and dyn_state are set up.
        >>> dyn_state.set_gravity(np.array([0, 0, 0, 0, 0, -9.81]))
        >>> # or dyn_state.set_Vdot0(np.array([0, 0, 0, 0, 0, 9.81]))  # Note that direction is reversed
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> gravity_torques = dyn_robot.compute_gravity_term(dyn_state)
        >>> print(gravity_torques)
        """
    def compute_inverse_dynamics(self, state: State_18) -> None:
        """
        Computes the inverse dynamics of the robot.
        
        This method calculates the joint torques required to achieve the given joint
        accelerations (`qddot`), considering the current joint positions (`q`) and
        velocities (`qdot`). The results are stored back into the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed joint torques.
            Hello. This is state.
        
        Notes
        -----
        `compute_forward_kinematics`, `compute_diff_forward_kinematics`, and
        `compute_2nd_diff_forward_kinematics` must be called in order before this function.
        
        Examples
        --------
        >>> # This example demonstrates the full sequence for inverse dynamics.
        >>> import rby1_sdk as rby
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> robot = rby.create_robot_a("localhost:50051")
        >>> robot.connect()
        >>> dyn_robot = robot.get_dynamics()
        >>> dyn_state = dyn_robot.make_state(
        ...     dyn_robot.get_link_names(), dyn_robot.get_joint_names()
        ... )
        >>> q = (np.random.rand(dyn_robot.get_dof()) - 0.5) * np.pi / 2
        >>> dyn_state.set_q(q)
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>>
        >>> # Perform kinematics calculations in order
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        >>>
        >>> # Compute inverse dynamics
        >>> dyn_robot.compute_inverse_dynamics(dyn_state)
        >>>
        >>> # Get the resulting torques
        >>> torques = dyn_state.get_tau()
        >>> with np.printoptions(precision=4, suppress=True):
        ...     print(f"Inverse dynamics torque (Nm): {torques}")
        """
    def compute_mass(self, state: State_18, target_link_index: int) -> float:
        """
        Computes the mass of a specific link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        float
            The mass of the specified link.
        """
    def compute_mass_matrix(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[18]], numpy.dtype[numpy.float64]]:
        """
        Computes the joint space mass matrix (inertia matrix) of the robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions.
        
        Returns
        -------
        numpy.ndarray
            The mass matrix (a square matrix of size DOF x DOF).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_mobility_diff_kinematics(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the forward differential kinematics for the mobile base.
        
        Calculates the linear and angular velocity of the mobile base from the current
        wheel velocities (`qdot`).
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including wheel velocities.
        
        Returns
        -------
        numpy.ndarray
            The resulting body velocity vector [w, vx, vy].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_18, linear_velocity: numpy.ndarray[tuple[typing.Literal[2], typing.Literal[1]], numpy.dtype[numpy.float64]], angular_velocity: float) -> None:
        """
        Computes the inverse differential kinematics for the mobile base.
        
        Calculates the required wheel velocities to achieve a desired linear and angular
        velocity of the mobile base. Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        linear_velocity : numpy.ndarray
            The desired linear velocity (x, y) [m/s].
        angular_velocity : float
            The desired angular velocity (yaw) [rad/s].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_18, body_velocity: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Computes the inverse differential kinematics for the mobile base from a body velocity vector.
        
        Calculates the required wheel velocities from a desired body velocity (twist).
        Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        body_velocity : numpy.ndarray
            The desired body velocity vector [w, vx, vy].
        """
    def compute_reflective_inertia(self, state: State_18, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the reflective inertia (task space inertia) of the target link with respect to the reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 reflective inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_space_jacobian(self, state: State_18, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[18]], numpy.dtype[numpy.float64]]:
        """
        Computes the space Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF space Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_total_inertial(self, state: State_18, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the total spatial inertia of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 total spatial inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_transformation(self, state: State_18, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        """
        Computes the transformation matrix from a reference link to a target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link (the 'from' frame).
        target_link_index : int
            The index of the target link (the 'to' frame).
        
        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix (SE(3)).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def detect_collisions_or_nearest_links(self, state: State_18, collision_threshold: int = 0) -> list[CollisionResult]:
        """
        Detects collisions or finds the nearest links in the robot model.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        collision_threshold : int, optional
            The minimum number of link pairs to return. The function first finds all
            colliding pairs. If the number of colliding pairs is less than this
            threshold, it will supplement the result with the nearest non-colliding
            link pairs until the total count reaches the threshold. The returned list
            is always sorted by distance. If set to 0, only actual collisions are
            returned. Default is 0.
        
        Returns
        -------
        list[rby1_sdk.dynamics.CollisionResult]
            A list of collision results, sorted by distance.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def get_base(self) -> Link:
        """
        Get the base link of the robot.
        
        Returns
        -------
        Link
            Base link.
        """
    def get_dof(self) -> int:
        """
        Get the number of degrees of freedom.
        
        Returns
        -------
        int
            Number of degrees of freedom.
        """
    def get_joint_names(self) -> list[str]:
        """
        Get the list of names of all joints.
        
        Returns
        -------
        list
            List of joint names.
        """
    def get_joint_property(self, state: State_18, getter: typing.Callable[[Joint], float]) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets a specific property for all joints using a provided getter function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        getter : callable
            A function that takes a joint object and returns a double value.
        
        Returns
        -------
        numpy.ndarray
            A vector containing the specified property for each joint.
        """
    def get_limit_q_lower(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower position limits (q) for each joint.
        """
    def get_limit_q_upper(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper position limits (q) for each joint.
        """
    def get_limit_qddot_lower(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qddot_upper(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qdot_lower(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower velocity limits (q_dot) for each joint.
        """
    def get_limit_qdot_upper(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper velocity limits (q_dot) for each joint.
        """
    def get_limit_torque(self, state: State_18) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the torque limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of torque limits for each joint.
        """
    @typing.overload
    def get_link(self, name: str) -> Link:
        """
        Get a link by name.
        
        Parameters
        ----------
        name : str
            Name of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    @typing.overload
    def get_link(self, state: State_18, index: int) -> Link:
        """
        Get a link by state and index.
        
        Parameters
        ----------
        state : State
            Current state of the robot.
        index : int
            Index of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of names of all links.
        
        Returns
        -------
        list
            List of link names.
        """
    def get_number_of_joints(self) -> int:
        """
        Get the number of joints.
        
        Returns
        -------
        int
            Number of joints.
        """
    def make_state(self, link_names: list[str], joint_names: list[str]) -> State_18:
        """
        Create a state from link and joint names.
        
        The state object is essential for using the robot dynamics functions.
        It stores the robot's state, its state vector (e.g., indices of joints and links), 
        and also serves as a cache for intermediate results in dynamics and
        kinematics calculations to optimize for speed.
        
        Parameters
        ----------
        link_names : list[str]
            List of link names.
        joint_names : list[str]
            List of joint names.
        
        Returns
        -------
        State
            A new state object.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>>
        >>> link_0 = rby_dyn.Link("link_0")
        >>> link_1 = rby_dyn.Link("link_1")
        >>> 
        >>> joint_0 = rby_dyn.Joint.make_revolute("joint_0", np.identity(4), np.array([0, 0, 1]))
        >>> joint_0.connect_links(link_0, link_1, np.identity(4), np.identity(4))
        >>> 
        >>> dyn_robot = rby_dyn.Robot(
        ...     rby_dyn.RobotConfiguration(name="sample_robot", base_link=link_0)
        ... )
        >>> 
        >>> dyn_state = dyn_robot.make_state(["link_0", "link_1"], ["joint_0"])
        >>> dyn_state.set_q(np.array([np.pi / 2]))  # Angle of joint_0 is 90 degrees
        >>> 
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Calculate transformation from link_0 to link_1
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)  # 0: link_0, 1: link_1
        >>> print(transform)
        """
class Robot_24:
    """
    Robot (DOF=24) dynamics model.
    
    Represents the dynamics of a robot with a given number of degrees of freedom.
    
    Attributes
    ----------
    base : Link
        Base link of the robot.
    link_names : list
        List of names of all links.
    joint_names : list
        List of names of all joints.
    """
    @staticmethod
    def count_joints(base_link: Link, include_fixed: bool = False) -> int:
        """
        count_joints(base_link, include_fixed=False)
        
        Counts the number of joints in a kinematic chain starting from a base link.
        
        Parameters
        ----------
        base_link : rby1_sdk.dynamics.Link
            The starting link of the kinematic chain.
        include_fixed : bool, optional
            Whether to include fixed joints in the count. Default is False.
        
        Returns
        -------
        int
            The total number of joints.
        """
    def __init__(self, robot_configuration: RobotConfiguration) -> None:
        """
        Construct a Robot instance.
        
        Parameters
        ----------
        robot_configuration : RobotConfiguration
            Configuration of the robot.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compute_2nd_diff_forward_kinematics(self, state: State_24) -> None:
        """
        Computes the second-order differential forward kinematics for each joint.
        
        This method calculates the body acceleration for each joint frame based on the
        current joint accelerations (`qddot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed body accelerations.
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        """
    def compute_body_jacobian(self, state: State_24, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[24]], numpy.dtype[numpy.float64]]:
        """
        Computes the body Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF body Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_body_velocity(self, state: State_24, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the relative body velocity (twist) of a target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x1 body velocity vector (twist).
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_24, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of a single target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_24, ref_link: int, target_links: list[int]) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the combined center of mass of multiple target links with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_links : list[int]
            A list of indices of the target links.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the combined center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_24, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the total center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_24, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[24]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of a single target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_24, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[24]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of the entire robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix for the whole robot.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_diff_forward_kinematics(self, state: State_24) -> None:
        """
        Computes the differential forward kinematics for each joint.
        
        This method calculates the body velocity (twist) for each joint frame based on
        the current joint velocities (`qdot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint velocities. This object
            will be updated with the computed body velocities.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        """
    def compute_forward_kinematics(self, state: State_24) -> None:
        """
        Computes the forward kinematics for each joint.
        
        This method calculates the transformation matrix from the base to each joint frame
        based on the current joint positions (`q`) in the state. The results are cached
        within the `state` object. This function must be called before other kinematics
        or dynamics calculations.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint positions. This object
            will be updated with the computed transformation matrices.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> # Assume dyn_robot is an initialized rby_dyn.Robot instance
        >>> # and dyn_state is a corresponding state object.
        >>> dyn_state.set_q(np.random.rand(dyn_robot.get_dof()))
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Now you can compute transformations, Jacobians, etc.
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)
        >>> print(transform)
        """
    def compute_gravity_term(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the gravity compensation term for the robot.
        
        This method calculates the joint torques required to counteract gravity at the
        current joint positions. The gravity vector must be set in the state object
        prior to calling this function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions and the gravity vector.
        
        Returns
        -------
        numpy.ndarray
            A vector of joint torques required to compensate for gravity.
        
        Notes
        -----
        - `compute_forward_kinematics` must be called before this function.
        - The gravity vector (spatial acceleration) must be set on the `state` object
          using `state.set_gravity()` or `state.set_Vdot0()`. For standard gravity along
          the negative Z-axis, the vector is `[0, 0, 0, 0, 0, -9.81]`.
        
        Examples
        --------
        >>> # Continuing from a previous example where dyn_robot and dyn_state are set up.
        >>> dyn_state.set_gravity(np.array([0, 0, 0, 0, 0, -9.81]))
        >>> # or dyn_state.set_Vdot0(np.array([0, 0, 0, 0, 0, 9.81]))  # Note that direction is reversed
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> gravity_torques = dyn_robot.compute_gravity_term(dyn_state)
        >>> print(gravity_torques)
        """
    def compute_inverse_dynamics(self, state: State_24) -> None:
        """
        Computes the inverse dynamics of the robot.
        
        This method calculates the joint torques required to achieve the given joint
        accelerations (`qddot`), considering the current joint positions (`q`) and
        velocities (`qdot`). The results are stored back into the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed joint torques.
            Hello. This is state.
        
        Notes
        -----
        `compute_forward_kinematics`, `compute_diff_forward_kinematics`, and
        `compute_2nd_diff_forward_kinematics` must be called in order before this function.
        
        Examples
        --------
        >>> # This example demonstrates the full sequence for inverse dynamics.
        >>> import rby1_sdk as rby
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> robot = rby.create_robot_a("localhost:50051")
        >>> robot.connect()
        >>> dyn_robot = robot.get_dynamics()
        >>> dyn_state = dyn_robot.make_state(
        ...     dyn_robot.get_link_names(), dyn_robot.get_joint_names()
        ... )
        >>> q = (np.random.rand(dyn_robot.get_dof()) - 0.5) * np.pi / 2
        >>> dyn_state.set_q(q)
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>>
        >>> # Perform kinematics calculations in order
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        >>>
        >>> # Compute inverse dynamics
        >>> dyn_robot.compute_inverse_dynamics(dyn_state)
        >>>
        >>> # Get the resulting torques
        >>> torques = dyn_state.get_tau()
        >>> with np.printoptions(precision=4, suppress=True):
        ...     print(f"Inverse dynamics torque (Nm): {torques}")
        """
    def compute_mass(self, state: State_24, target_link_index: int) -> float:
        """
        Computes the mass of a specific link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        float
            The mass of the specified link.
        """
    def compute_mass_matrix(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[24]], numpy.dtype[numpy.float64]]:
        """
        Computes the joint space mass matrix (inertia matrix) of the robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions.
        
        Returns
        -------
        numpy.ndarray
            The mass matrix (a square matrix of size DOF x DOF).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_mobility_diff_kinematics(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the forward differential kinematics for the mobile base.
        
        Calculates the linear and angular velocity of the mobile base from the current
        wheel velocities (`qdot`).
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including wheel velocities.
        
        Returns
        -------
        numpy.ndarray
            The resulting body velocity vector [w, vx, vy].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_24, linear_velocity: numpy.ndarray[tuple[typing.Literal[2], typing.Literal[1]], numpy.dtype[numpy.float64]], angular_velocity: float) -> None:
        """
        Computes the inverse differential kinematics for the mobile base.
        
        Calculates the required wheel velocities to achieve a desired linear and angular
        velocity of the mobile base. Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        linear_velocity : numpy.ndarray
            The desired linear velocity (x, y) [m/s].
        angular_velocity : float
            The desired angular velocity (yaw) [rad/s].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_24, body_velocity: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Computes the inverse differential kinematics for the mobile base from a body velocity vector.
        
        Calculates the required wheel velocities from a desired body velocity (twist).
        Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        body_velocity : numpy.ndarray
            The desired body velocity vector [w, vx, vy].
        """
    def compute_reflective_inertia(self, state: State_24, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the reflective inertia (task space inertia) of the target link with respect to the reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 reflective inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_space_jacobian(self, state: State_24, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[24]], numpy.dtype[numpy.float64]]:
        """
        Computes the space Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF space Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_total_inertial(self, state: State_24, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the total spatial inertia of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 total spatial inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_transformation(self, state: State_24, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        """
        Computes the transformation matrix from a reference link to a target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link (the 'from' frame).
        target_link_index : int
            The index of the target link (the 'to' frame).
        
        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix (SE(3)).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def detect_collisions_or_nearest_links(self, state: State_24, collision_threshold: int = 0) -> list[CollisionResult]:
        """
        Detects collisions or finds the nearest links in the robot model.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        collision_threshold : int, optional
            The minimum number of link pairs to return. The function first finds all
            colliding pairs. If the number of colliding pairs is less than this
            threshold, it will supplement the result with the nearest non-colliding
            link pairs until the total count reaches the threshold. The returned list
            is always sorted by distance. If set to 0, only actual collisions are
            returned. Default is 0.
        
        Returns
        -------
        list[rby1_sdk.dynamics.CollisionResult]
            A list of collision results, sorted by distance.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def get_base(self) -> Link:
        """
        Get the base link of the robot.
        
        Returns
        -------
        Link
            Base link.
        """
    def get_dof(self) -> int:
        """
        Get the number of degrees of freedom.
        
        Returns
        -------
        int
            Number of degrees of freedom.
        """
    def get_joint_names(self) -> list[str]:
        """
        Get the list of names of all joints.
        
        Returns
        -------
        list
            List of joint names.
        """
    def get_joint_property(self, state: State_24, getter: typing.Callable[[Joint], float]) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets a specific property for all joints using a provided getter function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        getter : callable
            A function that takes a joint object and returns a double value.
        
        Returns
        -------
        numpy.ndarray
            A vector containing the specified property for each joint.
        """
    def get_limit_q_lower(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower position limits (q) for each joint.
        """
    def get_limit_q_upper(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper position limits (q) for each joint.
        """
    def get_limit_qddot_lower(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qddot_upper(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qdot_lower(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower velocity limits (q_dot) for each joint.
        """
    def get_limit_qdot_upper(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper velocity limits (q_dot) for each joint.
        """
    def get_limit_torque(self, state: State_24) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the torque limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of torque limits for each joint.
        """
    @typing.overload
    def get_link(self, name: str) -> Link:
        """
        Get a link by name.
        
        Parameters
        ----------
        name : str
            Name of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    @typing.overload
    def get_link(self, state: State_24, index: int) -> Link:
        """
        Get a link by state and index.
        
        Parameters
        ----------
        state : State
            Current state of the robot.
        index : int
            Index of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of names of all links.
        
        Returns
        -------
        list
            List of link names.
        """
    def get_number_of_joints(self) -> int:
        """
        Get the number of joints.
        
        Returns
        -------
        int
            Number of joints.
        """
    def make_state(self, link_names: list[str], joint_names: list[str]) -> State_24:
        """
        Create a state from link and joint names.
        
        The state object is essential for using the robot dynamics functions.
        It stores the robot's state, its state vector (e.g., indices of joints and links), 
        and also serves as a cache for intermediate results in dynamics and
        kinematics calculations to optimize for speed.
        
        Parameters
        ----------
        link_names : list[str]
            List of link names.
        joint_names : list[str]
            List of joint names.
        
        Returns
        -------
        State
            A new state object.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>>
        >>> link_0 = rby_dyn.Link("link_0")
        >>> link_1 = rby_dyn.Link("link_1")
        >>> 
        >>> joint_0 = rby_dyn.Joint.make_revolute("joint_0", np.identity(4), np.array([0, 0, 1]))
        >>> joint_0.connect_links(link_0, link_1, np.identity(4), np.identity(4))
        >>> 
        >>> dyn_robot = rby_dyn.Robot(
        ...     rby_dyn.RobotConfiguration(name="sample_robot", base_link=link_0)
        ... )
        >>> 
        >>> dyn_state = dyn_robot.make_state(["link_0", "link_1"], ["joint_0"])
        >>> dyn_state.set_q(np.array([np.pi / 2]))  # Angle of joint_0 is 90 degrees
        >>> 
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Calculate transformation from link_0 to link_1
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)  # 0: link_0, 1: link_1
        >>> print(transform)
        """
class Robot_26:
    """
    Robot (DOF=26) dynamics model.
    
    Represents the dynamics of a robot with a given number of degrees of freedom.
    
    Attributes
    ----------
    base : Link
        Base link of the robot.
    link_names : list
        List of names of all links.
    joint_names : list
        List of names of all joints.
    """
    @staticmethod
    def count_joints(base_link: Link, include_fixed: bool = False) -> int:
        """
        count_joints(base_link, include_fixed=False)
        
        Counts the number of joints in a kinematic chain starting from a base link.
        
        Parameters
        ----------
        base_link : rby1_sdk.dynamics.Link
            The starting link of the kinematic chain.
        include_fixed : bool, optional
            Whether to include fixed joints in the count. Default is False.
        
        Returns
        -------
        int
            The total number of joints.
        """
    def __init__(self, robot_configuration: RobotConfiguration) -> None:
        """
        Construct a Robot instance.
        
        Parameters
        ----------
        robot_configuration : RobotConfiguration
            Configuration of the robot.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def compute_2nd_diff_forward_kinematics(self, state: State_26) -> None:
        """
        Computes the second-order differential forward kinematics for each joint.
        
        This method calculates the body acceleration for each joint frame based on the
        current joint accelerations (`qddot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed body accelerations.
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        """
    def compute_body_jacobian(self, state: State_26, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[26]], numpy.dtype[numpy.float64]]:
        """
        Computes the body Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF body Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_body_velocity(self, state: State_26, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the relative body velocity (twist) of a target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x1 body velocity vector (twist).
        
        Notes
        -----
        `compute_forward_kinematics` and `compute_diff_forward_kinematics` must be
        called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_26, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of a single target link with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_26, ref_link: int, target_links: list[int]) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the combined center of mass of multiple target links with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_links : list[int]
            A list of indices of the target links.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the combined center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass(self, state: State_26, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the center of mass of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3D position vector of the total center of mass.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_26, ref_link: int, target_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[26]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of a single target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        target_link : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    @typing.overload
    def compute_center_of_mass_jacobian(self, state: State_26, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[26]], numpy.dtype[numpy.float64]]:
        """
        Computes the Jacobian for the center of mass of the entire robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 3xDOF center of mass Jacobian matrix for the whole robot.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_diff_forward_kinematics(self, state: State_26) -> None:
        """
        Computes the differential forward kinematics for each joint.
        
        This method calculates the body velocity (twist) for each joint frame based on
        the current joint velocities (`qdot`) in the state. The results are cached
        within the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint velocities. This object
            will be updated with the computed body velocities.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        
        Examples
        --------
        >>> # Continuing from the previous example...
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        """
    def compute_forward_kinematics(self, state: State_26) -> None:
        """
        Computes the forward kinematics for each joint.
        
        This method calculates the transformation matrix from the base to each joint frame
        based on the current joint positions (`q`) in the state. The results are cached
        within the `state` object. This function must be called before other kinematics
        or dynamics calculations.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, which includes joint positions. This object
            will be updated with the computed transformation matrices.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> # Assume dyn_robot is an initialized rby_dyn.Robot instance
        >>> # and dyn_state is a corresponding state object.
        >>> dyn_state.set_q(np.random.rand(dyn_robot.get_dof()))
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Now you can compute transformations, Jacobians, etc.
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)
        >>> print(transform)
        """
    def compute_gravity_term(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the gravity compensation term for the robot.
        
        This method calculates the joint torques required to counteract gravity at the
        current joint positions. The gravity vector must be set in the state object
        prior to calling this function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions and the gravity vector.
        
        Returns
        -------
        numpy.ndarray
            A vector of joint torques required to compensate for gravity.
        
        Notes
        -----
        - `compute_forward_kinematics` must be called before this function.
        - The gravity vector (spatial acceleration) must be set on the `state` object
          using `state.set_gravity()` or `state.set_Vdot0()`. For standard gravity along
          the negative Z-axis, the vector is `[0, 0, 0, 0, 0, -9.81]`.
        
        Examples
        --------
        >>> # Continuing from a previous example where dyn_robot and dyn_state are set up.
        >>> dyn_state.set_gravity(np.array([0, 0, 0, 0, 0, -9.81]))
        >>> # or dyn_state.set_Vdot0(np.array([0, 0, 0, 0, 0, 9.81]))  # Note that direction is reversed
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> gravity_torques = dyn_robot.compute_gravity_term(dyn_state)
        >>> print(gravity_torques)
        """
    def compute_inverse_dynamics(self, state: State_26) -> None:
        """
        Computes the inverse dynamics of the robot.
        
        This method calculates the joint torques required to achieve the given joint
        accelerations (`qddot`), considering the current joint positions (`q`) and
        velocities (`qdot`). The results are stored back into the `state` object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions, velocities, and
            accelerations. This object will be updated with the computed joint torques.
            Hello. This is state.
        
        Notes
        -----
        `compute_forward_kinematics`, `compute_diff_forward_kinematics`, and
        `compute_2nd_diff_forward_kinematics` must be called in order before this function.
        
        Examples
        --------
        >>> # This example demonstrates the full sequence for inverse dynamics.
        >>> import rby1_sdk as rby
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>> robot = rby.create_robot_a("localhost:50051")
        >>> robot.connect()
        >>> dyn_robot = robot.get_dynamics()
        >>> dyn_state = dyn_robot.make_state(
        ...     dyn_robot.get_link_names(), dyn_robot.get_joint_names()
        ... )
        >>> q = (np.random.rand(dyn_robot.get_dof()) - 0.5) * np.pi / 2
        >>> dyn_state.set_q(q)
        >>> dyn_state.set_qdot(np.zeros(dyn_robot.get_dof()))
        >>> dyn_state.set_qddot(np.zeros(dyn_robot.get_dof()))
        >>>
        >>> # Perform kinematics calculations in order
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_diff_forward_kinematics(dyn_state)
        >>> dyn_robot.compute_2nd_diff_forward_kinematics(dyn_state)
        >>>
        >>> # Compute inverse dynamics
        >>> dyn_robot.compute_inverse_dynamics(dyn_state)
        >>>
        >>> # Get the resulting torques
        >>> torques = dyn_state.get_tau()
        >>> with np.printoptions(precision=4, suppress=True):
        ...     print(f"Inverse dynamics torque (Nm): {torques}")
        """
    def compute_mass(self, state: State_26, target_link_index: int) -> float:
        """
        Computes the mass of a specific link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        float
            The mass of the specified link.
        """
    def compute_mass_matrix(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[26]], numpy.dtype[numpy.float64]]:
        """
        Computes the joint space mass matrix (inertia matrix) of the robot.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including joint positions.
        
        Returns
        -------
        numpy.ndarray
            The mass matrix (a square matrix of size DOF x DOF).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_mobility_diff_kinematics(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Computes the forward differential kinematics for the mobile base.
        
        Calculates the linear and angular velocity of the mobile base from the current
        wheel velocities (`qdot`).
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot, including wheel velocities.
        
        Returns
        -------
        numpy.ndarray
            The resulting body velocity vector [w, vx, vy].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_26, linear_velocity: numpy.ndarray[tuple[typing.Literal[2], typing.Literal[1]], numpy.dtype[numpy.float64]], angular_velocity: float) -> None:
        """
        Computes the inverse differential kinematics for the mobile base.
        
        Calculates the required wheel velocities to achieve a desired linear and angular
        velocity of the mobile base. Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        linear_velocity : numpy.ndarray
            The desired linear velocity (x, y) [m/s].
        angular_velocity : float
            The desired angular velocity (yaw) [rad/s].
        """
    @typing.overload
    def compute_mobility_inverse_diff_kinematics(self, state: State_26, body_velocity: numpy.ndarray[tuple[typing.Literal[3], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Computes the inverse differential kinematics for the mobile base from a body velocity vector.
        
        Calculates the required wheel velocities from a desired body velocity (twist).
        Updates the `qdot` values in the state object.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The robot state object to be updated.
        body_velocity : numpy.ndarray
            The desired body velocity vector [w, vx, vy].
        """
    def compute_reflective_inertia(self, state: State_26, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the reflective inertia (task space inertia) of the target link with respect to the reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 reflective inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_space_jacobian(self, state: State_26, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[26]], numpy.dtype[numpy.float64]]:
        """
        Computes the space Jacobian for a target link relative to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link.
        target_link_index : int
            The index of the target link.
        
        Returns
        -------
        numpy.ndarray
            The 6xDOF space Jacobian matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_total_inertial(self, state: State_26, ref_link: int) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[6]], numpy.dtype[numpy.float64]]:
        """
        Computes the total spatial inertia of the entire robot with respect to a reference link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        ref_link : int
            The index of the reference link frame.
        
        Returns
        -------
        numpy.ndarray
            The 6x6 total spatial inertia matrix.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def compute_transformation(self, state: State_26, reference_link_index: int, target_link_index: int) -> numpy.ndarray[tuple[typing.Literal[4], typing.Literal[4]], numpy.dtype[numpy.float64]]:
        """
        Computes the transformation matrix from a reference link to a target link.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        reference_link_index : int
            The index of the reference link (the 'from' frame).
        target_link_index : int
            The index of the target link (the 'to' frame).
        
        Returns
        -------
        numpy.ndarray
            The 4x4 transformation matrix (SE(3)).
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def detect_collisions_or_nearest_links(self, state: State_26, collision_threshold: int = 0) -> list[CollisionResult]:
        """
        Detects collisions or finds the nearest links in the robot model.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        collision_threshold : int, optional
            The minimum number of link pairs to return. The function first finds all
            colliding pairs. If the number of colliding pairs is less than this
            threshold, it will supplement the result with the nearest non-colliding
            link pairs until the total count reaches the threshold. The returned list
            is always sorted by distance. If set to 0, only actual collisions are
            returned. Default is 0.
        
        Returns
        -------
        list[rby1_sdk.dynamics.CollisionResult]
            A list of collision results, sorted by distance.
        
        Notes
        -----
        `compute_forward_kinematics` must be called before this function.
        """
    def get_base(self) -> Link:
        """
        Get the base link of the robot.
        
        Returns
        -------
        Link
            Base link.
        """
    def get_dof(self) -> int:
        """
        Get the number of degrees of freedom.
        
        Returns
        -------
        int
            Number of degrees of freedom.
        """
    def get_joint_names(self) -> list[str]:
        """
        Get the list of names of all joints.
        
        Returns
        -------
        list
            List of joint names.
        """
    def get_joint_property(self, state: State_26, getter: typing.Callable[[Joint], float]) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets a specific property for all joints using a provided getter function.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        getter : callable
            A function that takes a joint object and returns a double value.
        
        Returns
        -------
        numpy.ndarray
            A vector containing the specified property for each joint.
        """
    def get_limit_q_lower(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower position limits (q) for each joint.
        """
    def get_limit_q_upper(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper position limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper position limits (q) for each joint.
        """
    def get_limit_qddot_lower(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qddot_upper(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper acceleration limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper acceleration limits (q_ddot) for each joint.
        """
    def get_limit_qdot_lower(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the lower velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of lower velocity limits (q_dot) for each joint.
        """
    def get_limit_qdot_upper(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the upper velocity limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of upper velocity limits (q_dot) for each joint.
        """
    def get_limit_torque(self, state: State_26) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Gets the torque limits for all joints.
        
        Parameters
        ----------
        state : rby1_sdk.dynamics.State
            The current state of the robot.
        
        Returns
        -------
        numpy.ndarray
            A vector of torque limits for each joint.
        """
    @typing.overload
    def get_link(self, name: str) -> Link:
        """
        Get a link by name.
        
        Parameters
        ----------
        name : str
            Name of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    @typing.overload
    def get_link(self, state: State_26, index: int) -> Link:
        """
        Get a link by state and index.
        
        Parameters
        ----------
        state : State
            Current state of the robot.
        index : int
            Index of the link.
        
        Returns
        -------
        Link, optional
            Link if found, None otherwise.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of names of all links.
        
        Returns
        -------
        list
            List of link names.
        """
    def get_number_of_joints(self) -> int:
        """
        Get the number of joints.
        
        Returns
        -------
        int
            Number of joints.
        """
    def make_state(self, link_names: list[str], joint_names: list[str]) -> State_26:
        """
        Create a state from link and joint names.
        
        The state object is essential for using the robot dynamics functions.
        It stores the robot's state, its state vector (e.g., indices of joints and links), 
        and also serves as a cache for intermediate results in dynamics and
        kinematics calculations to optimize for speed.
        
        Parameters
        ----------
        link_names : list[str]
            List of link names.
        joint_names : list[str]
            List of joint names.
        
        Returns
        -------
        State
            A new state object.
        
        Examples
        --------
        >>> import rby1_sdk.dynamics as rby_dyn
        >>> import numpy as np
        >>>
        >>> link_0 = rby_dyn.Link("link_0")
        >>> link_1 = rby_dyn.Link("link_1")
        >>> 
        >>> joint_0 = rby_dyn.Joint.make_revolute("joint_0", np.identity(4), np.array([0, 0, 1]))
        >>> joint_0.connect_links(link_0, link_1, np.identity(4), np.identity(4))
        >>> 
        >>> dyn_robot = rby_dyn.Robot(
        ...     rby_dyn.RobotConfiguration(name="sample_robot", base_link=link_0)
        ... )
        >>> 
        >>> dyn_state = dyn_robot.make_state(["link_0", "link_1"], ["joint_0"])
        >>> dyn_state.set_q(np.array([np.pi / 2]))  # Angle of joint_0 is 90 degrees
        >>> 
        >>> dyn_robot.compute_forward_kinematics(dyn_state)
        >>> # Calculate transformation from link_0 to link_1
        >>> transform = dyn_robot.compute_transformation(dyn_state, 0, 1)  # 0: link_0, 1: link_1
        >>> print(transform)
        """
class State:
    """
    Robot state for dynamics calculations.
    
    This class stores the state of the robot, including joint positions, velocities,
    accelerations, and torques. It also serves as a cache for intermediate results
    in dynamics and kinematics calculations to optimize performance.
    
    Attributes
    ----------
    base_link_idx : int
        Index of the base link.
    q : numpy.ndarray, shape (DOF,)
        Joint positions vector.
    qdot : numpy.ndarray, shape (DOF,)
        Joint velocities vector.
    qddot : numpy.ndarray, shape (DOF,)
        Joint accelerations vector.
    tau : numpy.ndarray, shape (DOF,)
        Joint torques vector (output of inverse dynamics).
    V0 : numpy.ndarray, shape (6,)
        Spatial velocity (twist) of the base link.
    Vdot0 : numpy.ndarray, shape (6,)
        Spatial acceleration of the base link (used to specify gravity).
        Note that `gravity = -Vdot0`.
    joint_names : list[str]
        List of joint names.
    link_names : list[str]
        List of link names.
    """
    q: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]
    qddot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]
    qdot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]
    tau: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_base_link_idx(self) -> int:
        """
        Get the index of the base link.
        
        Returns
        -------
        int
            Index of the base link.
        """
    def get_joint_names(self) -> list[str]:
        """
        Get the list of joint names associated with this state.
        
        Returns
        -------
        list[str]
            List of joint names.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of link names associated with this state.
        
        Returns
        -------
        list[str]
            List of link names.
        """
    def get_q(self) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint positions.
        
        Returns
        -------
        numpy.ndarray
            Joint positions vector.
        """
    def get_qddot(self) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint accelerations.
        
        Returns
        -------
        numpy.ndarray
            Joint accelerations vector.
        """
    def get_qdot(self) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint velocities.
        
        Returns
        -------
        numpy.ndarray
            Joint velocities vector.
        """
    def get_tau(self) -> numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint torques.
        
        Returns
        -------
        numpy.ndarray
            Joint torques vector.
        """
    def set_V0(self, V0: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial velocity of the base link.
        
        Parameters
        ----------
        V0 : numpy.ndarray
            6D spatial velocity vector (twist).
        """
    def set_Vdot0(self, Vdot0: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial acceleration of the base link.
        
        Parameters
        ----------
        Vdot0 : numpy.ndarray
            6D spatial acceleration vector.
        """
    def set_gravity(self, gravity: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the gravity vector. This is a convenience function that sets `Vdot0 = -gravity`.
        
        Parameters
        ----------
        gravity : numpy.ndarray
            6D gravity vector (e.g., `[0, 0, 0, 0, 0, -9.81]`).
        """
    def set_q(self, q: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint positions.
        
        Parameters
        ----------
        q : numpy.ndarray
            Joint positions vector.
        """
    def set_qddot(self, qddot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint accelerations.
        
        Parameters
        ----------
        qddot : numpy.ndarray
            Joint accelerations vector.
        """
    def set_qdot(self, qdot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint velocities.
        
        Parameters
        ----------
        qdot : numpy.ndarray
            Joint velocities vector.
        """
    def set_tau(self, tau: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint torques.
        
        Parameters
        ----------
        tau : numpy.ndarray
            Joint torques vector.
        """
    @property
    def V0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @V0.setter
    def V0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def Vdot0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @Vdot0.setter
    def Vdot0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def base_link_idx(self) -> int:
        ...
    @property
    def joint_names(self) -> list[str]:
        ...
    @property
    def link_names(self) -> list[str]:
        ...
class State_18:
    """
    Robot state (DOF=18) for dynamics calculations.
    
    This class stores the state of the robot, including joint positions, velocities,
    accelerations, and torques. It also serves as a cache for intermediate results
    in dynamics and kinematics calculations to optimize performance.
    
    Attributes
    ----------
    base_link_idx : int
        Index of the base link.
    q : numpy.ndarray, shape (DOF,)
        Joint positions vector.
    qdot : numpy.ndarray, shape (DOF,)
        Joint velocities vector.
    qddot : numpy.ndarray, shape (DOF,)
        Joint accelerations vector.
    tau : numpy.ndarray, shape (DOF,)
        Joint torques vector (output of inverse dynamics).
    V0 : numpy.ndarray, shape (6,)
        Spatial velocity (twist) of the base link.
    Vdot0 : numpy.ndarray, shape (6,)
        Spatial acceleration of the base link (used to specify gravity).
        Note that `gravity = -Vdot0`.
    joint_names : list[str]
        List of joint names.
    link_names : list[str]
        List of link names.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_base_link_idx(self) -> int:
        """
        Get the index of the base link.
        
        Returns
        -------
        int
            Index of the base link.
        """
    def get_joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(18)]:
        """
        Get the list of joint names associated with this state.
        
        Returns
        -------
        list[str]
            List of joint names.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of link names associated with this state.
        
        Returns
        -------
        list[str]
            List of link names.
        """
    def get_q(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint positions.
        
        Returns
        -------
        numpy.ndarray
            Joint positions vector.
        """
    def get_qddot(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint accelerations.
        
        Returns
        -------
        numpy.ndarray
            Joint accelerations vector.
        """
    def get_qdot(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint velocities.
        
        Returns
        -------
        numpy.ndarray
            Joint velocities vector.
        """
    def get_tau(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint torques.
        
        Returns
        -------
        numpy.ndarray
            Joint torques vector.
        """
    def set_V0(self, V0: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial velocity of the base link.
        
        Parameters
        ----------
        V0 : numpy.ndarray
            6D spatial velocity vector (twist).
        """
    def set_Vdot0(self, Vdot0: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial acceleration of the base link.
        
        Parameters
        ----------
        Vdot0 : numpy.ndarray
            6D spatial acceleration vector.
        """
    def set_gravity(self, gravity: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the gravity vector. This is a convenience function that sets `Vdot0 = -gravity`.
        
        Parameters
        ----------
        gravity : numpy.ndarray
            6D gravity vector (e.g., `[0, 0, 0, 0, 0, -9.81]`).
        """
    def set_q(self, q: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint positions.
        
        Parameters
        ----------
        q : numpy.ndarray
            Joint positions vector.
        """
    def set_qddot(self, qddot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint accelerations.
        
        Parameters
        ----------
        qddot : numpy.ndarray
            Joint accelerations vector.
        """
    def set_qdot(self, qdot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint velocities.
        
        Parameters
        ----------
        qdot : numpy.ndarray
            Joint velocities vector.
        """
    def set_tau(self, tau: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint torques.
        
        Parameters
        ----------
        tau : numpy.ndarray
            Joint torques vector.
        """
    @property
    def V0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @V0.setter
    def V0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def Vdot0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @Vdot0.setter
    def Vdot0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def base_link_idx(self) -> int:
        ...
    @property
    def joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(18)]:
        ...
    @property
    def link_names(self) -> list[str]:
        ...
    @property
    def q(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @q.setter
    def q(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qddot(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qddot.setter
    def qddot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qdot(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qdot.setter
    def qdot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def tau(self) -> numpy.ndarray[tuple[typing.Literal[18], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @tau.setter
    def tau(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
class State_24:
    """
    Robot state (DOF=24) for dynamics calculations.
    
    This class stores the state of the robot, including joint positions, velocities,
    accelerations, and torques. It also serves as a cache for intermediate results
    in dynamics and kinematics calculations to optimize performance.
    
    Attributes
    ----------
    base_link_idx : int
        Index of the base link.
    q : numpy.ndarray, shape (DOF,)
        Joint positions vector.
    qdot : numpy.ndarray, shape (DOF,)
        Joint velocities vector.
    qddot : numpy.ndarray, shape (DOF,)
        Joint accelerations vector.
    tau : numpy.ndarray, shape (DOF,)
        Joint torques vector (output of inverse dynamics).
    V0 : numpy.ndarray, shape (6,)
        Spatial velocity (twist) of the base link.
    Vdot0 : numpy.ndarray, shape (6,)
        Spatial acceleration of the base link (used to specify gravity).
        Note that `gravity = -Vdot0`.
    joint_names : list[str]
        List of joint names.
    link_names : list[str]
        List of link names.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_base_link_idx(self) -> int:
        """
        Get the index of the base link.
        
        Returns
        -------
        int
            Index of the base link.
        """
    def get_joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(24)]:
        """
        Get the list of joint names associated with this state.
        
        Returns
        -------
        list[str]
            List of joint names.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of link names associated with this state.
        
        Returns
        -------
        list[str]
            List of link names.
        """
    def get_q(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint positions.
        
        Returns
        -------
        numpy.ndarray
            Joint positions vector.
        """
    def get_qddot(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint accelerations.
        
        Returns
        -------
        numpy.ndarray
            Joint accelerations vector.
        """
    def get_qdot(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint velocities.
        
        Returns
        -------
        numpy.ndarray
            Joint velocities vector.
        """
    def get_tau(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint torques.
        
        Returns
        -------
        numpy.ndarray
            Joint torques vector.
        """
    def set_V0(self, V0: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial velocity of the base link.
        
        Parameters
        ----------
        V0 : numpy.ndarray
            6D spatial velocity vector (twist).
        """
    def set_Vdot0(self, Vdot0: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial acceleration of the base link.
        
        Parameters
        ----------
        Vdot0 : numpy.ndarray
            6D spatial acceleration vector.
        """
    def set_gravity(self, gravity: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the gravity vector. This is a convenience function that sets `Vdot0 = -gravity`.
        
        Parameters
        ----------
        gravity : numpy.ndarray
            6D gravity vector (e.g., `[0, 0, 0, 0, 0, -9.81]`).
        """
    def set_q(self, q: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint positions.
        
        Parameters
        ----------
        q : numpy.ndarray
            Joint positions vector.
        """
    def set_qddot(self, qddot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint accelerations.
        
        Parameters
        ----------
        qddot : numpy.ndarray
            Joint accelerations vector.
        """
    def set_qdot(self, qdot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint velocities.
        
        Parameters
        ----------
        qdot : numpy.ndarray
            Joint velocities vector.
        """
    def set_tau(self, tau: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint torques.
        
        Parameters
        ----------
        tau : numpy.ndarray
            Joint torques vector.
        """
    @property
    def V0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @V0.setter
    def V0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def Vdot0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @Vdot0.setter
    def Vdot0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def base_link_idx(self) -> int:
        ...
    @property
    def joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(24)]:
        ...
    @property
    def link_names(self) -> list[str]:
        ...
    @property
    def q(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @q.setter
    def q(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qddot(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qddot.setter
    def qddot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qdot(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qdot.setter
    def qdot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def tau(self) -> numpy.ndarray[tuple[typing.Literal[24], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @tau.setter
    def tau(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
class State_26:
    """
    Robot state (DOF=26) for dynamics calculations.
    
    This class stores the state of the robot, including joint positions, velocities,
    accelerations, and torques. It also serves as a cache for intermediate results
    in dynamics and kinematics calculations to optimize performance.
    
    Attributes
    ----------
    base_link_idx : int
        Index of the base link.
    q : numpy.ndarray, shape (DOF,)
        Joint positions vector.
    qdot : numpy.ndarray, shape (DOF,)
        Joint velocities vector.
    qddot : numpy.ndarray, shape (DOF,)
        Joint accelerations vector.
    tau : numpy.ndarray, shape (DOF,)
        Joint torques vector (output of inverse dynamics).
    V0 : numpy.ndarray, shape (6,)
        Spatial velocity (twist) of the base link.
    Vdot0 : numpy.ndarray, shape (6,)
        Spatial acceleration of the base link (used to specify gravity).
        Note that `gravity = -Vdot0`.
    joint_names : list[str]
        List of joint names.
    link_names : list[str]
        List of link names.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def get_base_link_idx(self) -> int:
        """
        Get the index of the base link.
        
        Returns
        -------
        int
            Index of the base link.
        """
    def get_joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(26)]:
        """
        Get the list of joint names associated with this state.
        
        Returns
        -------
        list[str]
            List of joint names.
        """
    def get_link_names(self) -> list[str]:
        """
        Get the list of link names associated with this state.
        
        Returns
        -------
        list[str]
            List of link names.
        """
    def get_q(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint positions.
        
        Returns
        -------
        numpy.ndarray
            Joint positions vector.
        """
    def get_qddot(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint accelerations.
        
        Returns
        -------
        numpy.ndarray
            Joint accelerations vector.
        """
    def get_qdot(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint velocities.
        
        Returns
        -------
        numpy.ndarray
            Joint velocities vector.
        """
    def get_tau(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        """
        Get the joint torques.
        
        Returns
        -------
        numpy.ndarray
            Joint torques vector.
        """
    def set_V0(self, V0: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial velocity of the base link.
        
        Parameters
        ----------
        V0 : numpy.ndarray
            6D spatial velocity vector (twist).
        """
    def set_Vdot0(self, Vdot0: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the spatial acceleration of the base link.
        
        Parameters
        ----------
        Vdot0 : numpy.ndarray
            6D spatial acceleration vector.
        """
    def set_gravity(self, gravity: numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the gravity vector. This is a convenience function that sets `Vdot0 = -gravity`.
        
        Parameters
        ----------
        gravity : numpy.ndarray
            6D gravity vector (e.g., `[0, 0, 0, 0, 0, -9.81]`).
        """
    def set_q(self, q: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint positions.
        
        Parameters
        ----------
        q : numpy.ndarray
            Joint positions vector.
        """
    def set_qddot(self, qddot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint accelerations.
        
        Parameters
        ----------
        qddot : numpy.ndarray
            Joint accelerations vector.
        """
    def set_qdot(self, qdot: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint velocities.
        
        Parameters
        ----------
        qdot : numpy.ndarray
            Joint velocities vector.
        """
    def set_tau(self, tau: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        """
        Set the joint torques.
        
        Parameters
        ----------
        tau : numpy.ndarray
            Joint torques vector.
        """
    @property
    def V0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @V0.setter
    def V0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def Vdot0(self) -> numpy.ndarray[tuple[typing.Literal[6], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @Vdot0.setter
    def Vdot0(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def base_link_idx(self) -> int:
        ...
    @property
    def joint_names(self) -> typing.Annotated[list[str], pybind11_stubgen.typing_ext.FixedSize(26)]:
        ...
    @property
    def link_names(self) -> list[str]:
        ...
    @property
    def q(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @q.setter
    def q(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qddot(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qddot.setter
    def qddot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def qdot(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @qdot.setter
    def qdot(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
    @property
    def tau(self) -> numpy.ndarray[tuple[typing.Literal[26], typing.Literal[1]], numpy.dtype[numpy.float64]]:
        ...
    @tau.setter
    def tau(self, arg1: numpy.ndarray[tuple[M, typing.Literal[1]], numpy.dtype[numpy.float64]]) -> None:
        ...
def load_robot_from_urdf(path: str, base_link_name: str) -> RobotConfiguration:
    """
    Load a robot model from a URDF file.
    
    This function reads a URDF file from the given path and constructs
    a robot model based on its description.
    
    Parameters
    ----------
    path : str
        File system path to the URDF file.
    base_link_name : str
        Name of the base link in the URDF. This will be used as the
        reference link of the robot.
    
    Returns
    -------
    Robot
        The loaded robot model instance.
    
    Examples
    --------
    >>> robot = load_robot_from_urdf("path/to/robot.urdf", "base_link")
    """
def load_robot_from_urdf_data(model: str, base_link_name: str) -> RobotConfiguration:
    """
    Load a robot model from URDF data string.
    
    This function parses URDF XML content directly from a string and
    constructs a robot model. It is useful when URDF data is already
    loaded in memory and does not need to be read from a file.
    
    Parameters
    ----------
    model : str
        URDF XML data as a string.
    base_link_name : str
        Name of the base link in the URDF. This will be used as the
        reference link of the robot.
    
    Returns
    -------
    Robot
        The loaded robot model instance.
    
    Examples
    --------
    >>> with open("robot.urdf") as f:
    ...     urdf_data = f.read()
    >>> robot = load_robot_from_urdf_data(urdf_data, "base_link")
    """
