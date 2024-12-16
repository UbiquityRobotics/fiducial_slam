from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Find the path to the stag_detect package
    stag_detect_share_dir = get_package_share_directory('stag_detect')
    single_yaml_path = os.path.join(stag_detect_share_dir, 'cfg', 'single.yaml')

    # Declare the launch argument
    fiducial_transform_topic_arg = DeclareLaunchArgument(
        'fiducial_transform_topic',
        default_value='/fiducial_transforms',
        description='Fiducial transform topic override'
    )

    # Node definition
    stag_detect_node = Node(
        package='stag_detect',
        executable='stag_detect',
        name='stag_detect',
        output='screen',
        parameters=[
            {'marker_size': 0.18},
            {'stag_library': 11},
            single_yaml_path  # Load parameters from YAML file
        ],
        remappings=[
            ('stag_ros/markers_array', LaunchConfiguration('fiducial_transform_topic'))
        ],
    )

    return LaunchDescription([
        fiducial_transform_topic_arg,
        stag_detect_node,
    ])
