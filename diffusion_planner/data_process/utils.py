"""
Module: Coordination Transformation Functions and Numpy-Tensor Transformation
Description: This module contains functions for transforming the coordination to ego-centric coordination and Numpy-Tensor transformation.

Categories:
    1. Ego, agent, static coordination transformation
    2. Map coordination transformation
    3. Numpy-Tensor transformation
"""

import numpy as np
import torch

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

# =====================
# 1. Ego, agent, static coordination transformation
# =====================
def _local_to_local_transforms(global_states1, global_states2):
    """
    Converts the global_states1' local coordinates to global_states2's local coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(global_states2)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states1)

    transforms = np.matmul(local_xform_inv, transforms)

    return transforms

def _state_se2_array_to_transform_matrix(input_data):


    x: float = float(input_data[0])
    y: float = float(input_data[1])
    h: float = float(input_data[2])

    cosine = np.cos(h)
    sine = np.sin(h)

    return np.array(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]]
    )

def _state_se2_array_to_transform_matrix_batch(input_data):

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = np.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            np.cos(input_data[:, 2]),
            np.sin(input_data[:, 2]),
            np.ones_like(input_data[:, 0]),
        )
    )

    # See below for reshaping example
    reshaping_array = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 array, and then reshapes to a Nx3x3 array
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_array).reshape(-1, 3, 3)

def _transform_matrix_to_state_se2_array_batch(input_data):
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 array of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted array.
    """

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = np.arctan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result

def _global_state_se2_array_to_local(
    global_states, local_state
):
    """
    Transforms the StateSE2 in array from to the frame of reference in local_frame.

    :param global_states: A array of Nx3, where the columns are [x, y, heading].
    :param local_state: A array of [x, y, h] of the frame to which to transform.
    :return: The transformed coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(local_state)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states)

    transforms = np.matmul(local_xform_inv, transforms)

    output = _transform_matrix_to_state_se2_array_batch(transforms)

    return output

def _global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * np.cos(anchor_heading) + velocity[:, 1] * np.sin(anchor_heading)
    velocity_y = velocity[:, 1] * np.cos(anchor_heading) - velocity[:, 0] * np.sin(anchor_heading)

    return np.stack([velocity_x, velocity_y], axis=-1)



def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent or ego history to ego-centric coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = np.array(
        [
            float(ego_state[EgoInternalIndex.x()]),
            float(ego_state[EgoInternalIndex.y()]),
            float(ego_state[EgoInternalIndex.heading()]),
        ],
        dtype=np.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transforms = _local_to_local_transforms(agent_global_poses, ego_pose)
        transformed_poses = _transform_matrix_to_state_se2_array_batch(transforms)
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0]
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1]
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2]

        # local vel,acc to local
        agent_local_vel = agent_state[:, [EgoInternalIndex.vx(), EgoInternalIndex.vy()]]
        agent_local_acc = agent_state[:, [EgoInternalIndex.ax(), EgoInternalIndex.ay()]]
        agent_local_vel = np.expand_dims(np.concatenate((agent_local_vel, np.zeros((agent_local_vel.shape[0], 1))), axis=-1), axis=-1)
        agent_local_acc = np.expand_dims(np.concatenate((agent_local_acc, np.zeros((agent_local_acc.shape[0], 1))), axis=-1), axis=-1)
        transformed_vel = np.matmul(transforms, agent_local_vel).squeeze(axis=-1)
        transformed_acc = np.matmul(transforms, agent_local_acc).squeeze(axis=-1)
        agent_state[:, EgoInternalIndex.vx()] = transformed_vel[:, 0]
        agent_state[:, EgoInternalIndex.vy()] = transformed_vel[:, 1]
        agent_state[:, EgoInternalIndex.ax()] = transformed_acc[:, 0]
        agent_state[:, EgoInternalIndex.ay()] = transformed_acc[:, 1]
    elif agent_type == 'agent':
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = _global_state_se2_array_to_local(agent_global_poses, ego_pose)
        transformed_velocities = _global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0]
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1]
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2]
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0]
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1]
    elif agent_type == 'static':
        agent_global_poses = agent_state[:, [0, 1, 2]]
        transformed_poses = _global_state_se2_array_to_local(agent_global_poses, ego_pose)
        agent_state[:, 0] = transformed_poses[:, 0]
        agent_state[:, 1] = transformed_poses[:, 1]
        agent_state[:, 2] = transformed_poses[:, 2]
    elif agent_type == 'pos_heading_only':
        agent_global_poses = agent_state
        transformed_poses = _global_state_se2_array_to_local(agent_global_poses, ego_pose)
        agent_state = transformed_poses
    return agent_state


# =====================
# 2. Map coordination transformation
# =====================
def coordinates_to_local_frame(
    coords, anchor_state, precision = None
):
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <np.array: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate array. If None, then it will be inferred from the input precisions.
    :return: <np.array: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = _state_se2_array_to_transform_matrix(anchor_state)
    transform = np.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = np.pad(coords, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = np.matmul(transform, coords.T)

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.T
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords,
    avails,
    anchor_state,
    output_precision = np.float32,
):
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <np.array: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <np.array: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output array.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """


    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords, anchor_state, precision=np.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.astype(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords


# =====================
# 3. Numpy-Tensor transformation
# =====================
def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:
            tensor_data[k] = torch.tensor(v, dtype=torch.bool).unsqueeze(0).to(device)
        else:
            tensor_data[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)

    return tensor_data

# =====================
# 4. scenario map filter
# =====================
def get_scenario_map():
    scenario_map = {
        'accelerating_at_crosswalk': [15.0, -3.0],
        'accelerating_at_stop_sign': [15.0, -3.0],
        'accelerating_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'accelerating_at_traffic_light': [15.0, -3.0],
        'accelerating_at_traffic_light_with_lead': [15.0, -3.0],
        'accelerating_at_traffic_light_without_lead': [15.0, -3.0],
        'behind_bike': [15.0, -3.0],
        'behind_long_vehicle': [15.0, -3.0],
        'behind_pedestrian_on_driveable': [15.0, -3.0],
        'behind_pedestrian_on_pickup_dropoff': [15.0, -3.0],
        'changing_lane': [15.0, -3.0],
        'changing_lane_to_left': [15.0, -3.0],
        'changing_lane_to_right': [15.0, -3.0],
        'changing_lane_with_lead': [15.0, -3.0],
        'changing_lane_with_trail': [15.0, -3.0],
        'crossed_by_bike': [15.0, -3.0],
        'crossed_by_vehicle': [15.0, -3.0],
        'following_lane_with_lead': [15.0, -3.0],
        'following_lane_with_slow_lead': [15.0, -3.0],
        'following_lane_without_lead': [15.0, -3.0],
        'high_lateral_acceleration': [15.0, -3.0],
        'high_magnitude_jerk': [15.0, -3.0],
        'high_magnitude_speed': [15.0, -3.0],
        'low_magnitude_speed': [15.0, -3.0],
        'medium_magnitude_speed': [15.0, -3.0],
        'near_barrier_on_driveable': [15.0, -3.0],
        'near_construction_zone_sign': [15.0, -3.0],
        'near_high_speed_vehicle': [15.0, -3.0],
        'near_long_vehicle': [15.0, -3.0],
        'near_multiple_bikes': [15.0, -3.0],
        'near_multiple_pedestrians': [15.0, -3.0],
        'near_multiple_vehicles': [15.0, -3.0],
        'near_pedestrian_at_pickup_dropoff': [15.0, -3.0],
        'near_pedestrian_on_crosswalk': [15.0, -3.0],
        'near_pedestrian_on_crosswalk_with_ego': [15.0, -3.0],
        'near_trafficcone_on_driveable': [15.0, -3.0],
        'on_all_way_stop_intersection': [15.0, -3.0],
        'on_carpark': [15.0, -3.0],
        'on_intersection': [15.0, -3.0],
        'on_pickup_dropoff': [15.0, -3.0],
        'on_stopline_crosswalk': [15.0, -3.0],
        'on_stopline_stop_sign': [15.0, -3.0],
        'on_stopline_traffic_light': [15.0, -3.0],
        'on_traffic_light_intersection': [15.0, -3.0],
        'starting_high_speed_turn': [15.0, -3.0],
        'starting_left_turn': [15.0, -3.0],
        'starting_low_speed_turn': [15.0, -3.0],
        'starting_protected_cross_turn': [15.0, -3.0],
        'starting_protected_noncross_turn': [15.0, -3.0],
        'starting_right_turn': [15.0, -3.0],
        'starting_straight_stop_sign_intersection_traversal': [15.0, -3.0],
        'starting_straight_traffic_light_intersection_traversal': [15.0, -3.0],
        'starting_u_turn': [15.0, -3.0],
        'starting_unprotected_cross_turn': [15.0, -3.0],
        'starting_unprotected_noncross_turn': [15.0, -3.0],
        'stationary': [15.0, -3.0],
        'stationary_at_crosswalk': [15.0, -3.0],
        'stationary_at_traffic_light_with_lead': [15.0, -3.0],
        'stationary_at_traffic_light_without_lead': [15.0, -3.0],
        'stationary_in_traffic': [15.0, -3.0],
        'stopping_at_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_no_crosswalk': [15.0, -3.0],
        'stopping_at_stop_sign_with_lead': [15.0, -3.0],
        'stopping_at_stop_sign_without_lead': [15.0, -3.0],
        'stopping_at_traffic_light_with_lead': [15.0, -3.0],
        'stopping_at_traffic_light_without_lead': [15.0, -3.0],
        'stopping_with_lead': [15.0, -3.0],
        'traversing_crosswalk': [15.0, -3.0],
        'traversing_intersection': [15.0, -3.0],
        'traversing_narrow_lane': [15.0, -3.0],
        'traversing_pickup_dropoff': [15.0, -3.0],
        'traversing_traffic_light_intersection': [15.0, -3.0],
        'waiting_for_pedestrian_to_cross': [15.0, -3.0]
    }

    return scenario_map

def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None):
    scenario_types = [
        'starting_left_turn',
        'starting_right_turn',
        'starting_straight_traffic_light_intersection_traversal',
        'stopping_with_lead',
        'high_lateral_acceleration',
        'high_magnitude_speed',
        'low_magnitude_speed',
        'traversing_pickup_dropoff',
        'waiting_for_pedestrian_to_cross',
        'behind_long_vehicle',
        'stationary_in_traffic',
        'near_multiple_vehicles',
        'changing_lane',
        'following_lane_with_lead',
    ]

    scenario_tokens = None              # List of scenario tokens to include
    log_names = None                     # Filter scenarios by log names
    map_names = None                     # Filter scenarios by map names

    num_scenarios_per_type = num_scenarios_per_type    # Number of scenarios per type
    limit_total_scenarios = limit_total_scenarios       # Limit total scenarios (float = fraction, int = num) - this filter can be applied on top of num_scenarios_per_type
    timestamp_threshold_s = None          # Filter scenarios to ensure scenarios have more than `timestamp_threshold_s` seconds between their initial lidar timestamps
    ego_displacement_minimum_m = None    # Whether to remove scenarios where the ego moves less than a certain amount

    expand_scenarios = False           # Whether to expand multi-sample scenarios to multiple single-sample scenarios
    remove_invalid_goals = True         # Whether to remove scenarios where the mission goal is invalid
    shuffle = False                      # Whether to shuffle the scenarios

    ego_start_speed_threshold = None     # Limit to scenarios where the ego reaches a certain speed from below
    ego_stop_speed_threshold = None      # Limit to scenarios where the ego reaches a certain speed from above
    speed_noise_tolerance = None         # Value at or below which a speed change between two timepoints should be ignored as noise.

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance

########## Network input features ##########
def _extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a tensor as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a tensor.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types

def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects.
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = _extract_agent_tensor(past_tracked_objects[i], track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def sampled_tracked_objects_to_tensor(tracked_objects):
    """
    Tensorizes the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each tensor as described in `_extract_agent_tensor()`.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects.
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    tensorized, track_token_ids, agent_types = _extract_agent_tensor(tracked_objects, track_token_ids, object_types)
    output.append(tensorized)
    output_types.append(agent_types)

    return output, output_types
