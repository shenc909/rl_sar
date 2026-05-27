# Bridge: republish the Unitree Go2 /lowstate motor feedback as a standard
# sensor_msgs/JointState on /joint_states. The Go2 SDK only publishes LowState
# (no JointState), but robot_state_publisher needs JointState to broadcast the
# moving joint TF that robot_self_filter relies on (see go2_self_filter.launch.py).

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import JointState
from unitree_go.msg import LowState

# Unitree Go2 motor_state index order. Matches policy/go2/base.yaml joint_names
# (identity joint_mapping), so motor_state[i] corresponds to JOINT_NAMES[i].
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]


class LowStateToJointStates(Node):
    def __init__(self):
        super().__init__("low_state_to_joint_states")

        self.declare_parameter("lowstate_topic", "/lowstate")
        self.declare_parameter("joint_states_topic", "/joint_states")
        lowstate_topic = self.get_parameter("lowstate_topic").value
        joint_states_topic = self.get_parameter("joint_states_topic").value

        self.num_dofs = len(JOINT_NAMES)
        self.publisher = self.create_publisher(JointState, joint_states_topic, 10)
        self.subscription = self.create_subscription(
            LowState, lowstate_topic, self.low_state_callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Bridging {lowstate_topic} (LowState) -> {joint_states_topic} (JointState)"
        )

    def low_state_callback(self, msg: LowState):
        joint_state = JointState()
        joint_state.header.stamp = self.get_clock().now().to_msg()
        joint_state.name = JOINT_NAMES
        joint_state.position = [float(msg.motor_state[i].q) for i in range(self.num_dofs)]
        joint_state.velocity = [float(msg.motor_state[i].dq) for i in range(self.num_dofs)]
        joint_state.effort = [float(msg.motor_state[i].tau_est) for i in range(self.num_dofs)]
        self.publisher.publish(joint_state)


def main(args=None):
    rclpy.init(args=args)
    node = LowStateToJointStates()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
