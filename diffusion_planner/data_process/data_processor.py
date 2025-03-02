import numpy as np
import argparse
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import Task

from tqdm import tqdm

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
agent_past_process,
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.utils import (convert_to_model_inputs,
get_scenario_map,
get_filter_parameters,
sampled_tracked_objects_to_tensor_list,
sampled_tracked_objects_to_tensor
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative,_global_state_se2_array_to_local,_global_velocity_to_local
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states
)
import os, torch

from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
class DataProcessor(object):
    def __init__(self, scenarios, device):

        self.num_agents = 30 #[int]
        self.num_static = 30 #[int]
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 80 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'LEFT_BOUNDARY': 10, 'RIGHT_BOUNDARY': 10, 'ROUTE_LANES': 10} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'LEFT_BOUNDARY': 50, 'RIGHT_BOUNDARY': 50, 'ROUTE_LANES': 50} # maximum number of points per feature to extract per feature layer.
        self._vehicle_parameters = get_pacifica_parameters()

        self._scenarios = scenarios
        self.past_time_horizon = 2 #[seconds]
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8 #[seconds]
        self.num_future_poses = self.future_time_horizon * 10
        self.device = device

    def observation_adapter(self, history_buffer, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):

        '''
        ego
        '''
        ego_agent_past = None # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], dtype=np.float64)

        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)

        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids
        )
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius, traffic_light_data
        )
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features,
                                    self._max_elements, self._max_points)


        data = {"neighbor_agents_past": neighbor_agents_past[:, -21:],
                "ego_current_state": np.array([0., 0., 1. ,0.], dtype=np.float32), # ego centric x, y, cos, sin
                "static_objects": static_objects}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        return data

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state

        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_poses, time_horizon=self.past_time_horizon
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor


    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_horizon, num_samples=self.num_past_poses
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
            sampled_tracked_objects_to_tensor_list(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def add_agent_traj_noise(self, origin_traj: np.array, sde:SDE):
       device = self.device
       P, T, D = origin_traj.shape # P: agent num， T: frame nums, D: dim
       assert D == 4  #x, y, cos, sin
       origin_traj_tensor = torch.tensor(origin_traj, dtype=torch.float32).to(device)

       t = torch.rand(origin_traj.shape[0], device = device)
       mean, std = sde.marginal_prob(origin_traj_tensor, t)

       noisy_traj = torch.zeros_like(mean).to(device)
       noisy_traj = mean + std*noisy_traj
       theta_norm = torch.sqrt(noisy_traj[:,:,2]**2 + noisy_traj[:,:,3]**2 + 1e-6).to(device)
       noisy_traj[:,:,2] /= theta_norm
       noisy_traj[:,:,3] /= theta_norm

       return noisy_traj, t #tensor



    def work(self, save_dir, debug=False):
      for scenario in tqdm(self._scenarios):
         map_name = scenario._map_name
         token = scenario.token
         self.scenario = scenario
         self.map_api = scenario.map_api
         sde = VPSDE_linear()
         """
         ego
         """
         ego_state = self.scenario.initial_ego_state
         ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
         anchor_ego_state = np.array([ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading], \
                                      dtype=np.float64)
         """
         ego future traj
         """
         future_ego_states = self.scenario.get_ego_future_trajectory(iteration=0, num_samples=self.num_future_poses, time_horizon= self.future_time_horizon)
         future_ego_states_numpy = np.array([(ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading, \
                            ego_state.dynamic_car_state.rear_axle_velocity_2d.x, ego_state.dynamic_car_state.rear_axle_velocity_2d.y,\
                            ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y ) for ego_state in future_ego_states], dtype = np.float64)
         local_future_ego_states = convert_absolute_quantities_to_relative(future_ego_states_numpy, anchor_ego_state,'ego')

         ego_current_state_array = np.array([(ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading, \
                  ego_state.dynamic_car_state.rear_axle_velocity_2d.x, ego_state.dynamic_car_state.rear_axle_velocity_2d.y,\
                  ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, ego_state.dynamic_car_state.rear_axle_acceleration_2d.y )], dtype = np.float64)
         local_ego_current_state_array = convert_absolute_quantities_to_relative(ego_current_state_array, anchor_ego_state,'ego')
         future_ego_states_numpy =np.concatenate((local_ego_current_state_array,future_ego_states_numpy),axis = 0)
         future_ego_noise_traj =np.array([ np.concatenate((future_ego_states_numpy[:, :2], np.cos(future_ego_states_numpy[:, 3:4]), np.sin(future_ego_states_numpy[:, 3:4])), \
                                                axis =1)])
         future_ego_noise_traj, ego_diffusion_time = self.add_agent_traj_noise(future_ego_noise_traj, sde)

         """
         neighbor agents
         """
         neighbors = self.scenario.initial_tracked_objects
         neighbor_past = list(self.scenario.get_past_tracked_objects(iteration= 0, time_horizon=1.0))
         tracked_objects, tracked_objects_types, tracked_obj_tokens = sampled_tracked_objects_to_tensor(neighbors.tracked_objects)
        #  tracked_objects_past, _ = sampled_tracked_objects_to_array_list(neighbor_past)
         static_objects, static_objects_types =sampled_static_objects_to_array_list(neighbor_past[-1])
         tracked_objects_list = []
         for tracked_object in tracked_objects:
            tracked_objects_list.append(tracked_object)
         _, neighbor_past, selected_indices, static_objects = agent_past_process(None, tracked_objects, tracked_objects_types, self.num_agents,\
                                                                  static_objects, static_objects_types, \
                                                                  self.num_static, self.max_ped_bike, anchor_ego_state)
         """
         selected neighbor future agents
         """
         objects_tokens = [tracked_obj_tokens[key] for key in selected_indices]

         neighbor_future = list(self.scenario.get_future_tracked_objects(iteration= 0, time_horizon=self.future_time_horizon, num_samples =self.num_future_poses))
         slected_objs_future_traj = np.full((len(neighbor_future), len(neighbor_past), 11), np.nan, dtype=np.float64)
         for frame_id in range(len(neighbor_future)):
            tracked_objects = neighbor_future[frame_id]
            current_frame_tracked_objects = {tracked_object.track_token: tracked_object for tracked_object in tracked_objects.tracked_objects}
            for i in range(len(objects_tokens)):
              if objects_tokens[i] in current_frame_tracked_objects.keys():
                agent_state = current_frame_tracked_objects[objects_tokens[i]]

                agent_global_poses = np.array([[agent_state.center.x, agent_state.center.y, agent_state.center.heading]])
                agent_global_velocities = np.array([[agent_state.velocity.x, agent_state.velocity.y]])
                transformed_poses = _global_state_se2_array_to_local(agent_global_poses, anchor_ego_state)
                transformed_velocities = _global_velocity_to_local(agent_global_velocities, anchor_ego_state[-1])
                local_agent_state = np.zeros((1, 11))
                local_agent_state[:, 0] = transformed_poses[:, 0]
                local_agent_state[:, 1] = transformed_poses[:, 1]
                local_agent_state[:, 2] = np.cos(transformed_poses[:, 2])
                local_agent_state[:, 3] = np.sin(transformed_poses[:, 2])
                local_agent_state[:, 4] = transformed_velocities[:, 0]
                local_agent_state[:, 5] = transformed_velocities[:, 1]
                local_agent_state[:, 6] = agent_state.box.width
                local_agent_state[:, 7] = agent_state.box.length
                if agent_state.tracked_object_type == TrackedObjectType.VEHICLE:
                  local_agent_state[:, 8:] = [1, 0, 0]  # Mark as VEHICLE
                elif agent_state.tracked_object_type == TrackedObjectType.PEDESTRIAN:
                  local_agent_state[:, 8:] = [0, 1, 0]  # Mark as PEDESTRIAN
                else:  # TrackedObjectType.BICYCLE
                  local_agent_state[:, 8:] = [0, 0, 1]  # Mark as BICYCLE
                slected_objs_future_traj[frame_id, i, :] = local_agent_state
         slected_objs_future_traj = slected_objs_future_traj.transpose(1, 0, 2) # (num_neghbor, current_frame + future_frames , data_dim)
         slected_objs_future_traj = np.concatenate((neighbor_past, slected_objs_future_traj),axis = 1)
         slected_objs_future_traj = np.concatenate((
            slected_objs_future_traj[:,:, :2],
            np.cos(slected_objs_future_traj[:,:, 2:3]),
            np.sin(slected_objs_future_traj[:,:, 2:3])
            ), axis=2)
         """
         map
         """
         route_roadblocks_ids = self.scenario.get_route_roadblock_ids()

         route_roadblocks_ids =route_roadblock_correction(
                              ego_state, self.map_api, route_roadblocks_ids
                              )
         traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(iteration= 0)
         coords, traffic_light_data, speed_limit, lane_route =get_neighbor_vector_set_map(
                      self.map_api, self._map_features, ego_coords, self._radius, traffic_light_data)
         vector_map = map_process(route_roadblocks_ids, anchor_ego_state, coords, traffic_light_data, speed_limit, lane_route, self._map_features,
                                      self._max_elements, self._max_points)

         """
         add noise to agent traj
         """
         agent_noisy_traj, agent_diffusion_time = self.add_agent_traj_noise(slected_objs_future_traj, sde)
         pad_ego_agent_noisy_traj = torch.concat((future_ego_noise_traj, agent_noisy_traj), dim=0)
         pad_ego_agent_diffusion_time = torch.concat((ego_diffusion_time, agent_diffusion_time), dim=0)

         """
         data convert
         """
         data = {"neighbor_agents_past": neighbor_past[:, -21:],
        "ego_current_state": np.array([0., 0., 1. ,0.], dtype=np.float32), # ego centric x, y, cos, sin
        "static_objects": static_objects,
        'neighbor_future_traj':slected_objs_future_traj,
        'ego_future_traj': local_future_ego_states,
        'sampled_trajectories': pad_ego_agent_noisy_traj,
        'diffusion_time': pad_ego_agent_diffusion_time
        }
         data.update(vector_map)
         data = convert_to_model_inputs(data, self.device)

        # generate file name by log name + token + scenario type
         file_name = f"{self.scenario.log_name}_{self.scenario.token}_{self.scenario.scenario_type}.pt"
         file_name = file_name.replace(":", "_").replace("/", "_").replace(" ", "_")

         os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
         file_path = os.path.join(save_dir, file_name)
         torch.save(data, file_path)

def main(data_path, save_path, total_scenarios=100000, map_version="nuplan-maps-v1.0"):
    map_path = "/share/data_cold/open_data/nuplan/maps"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(data_path, map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=128)
    scenario_filter = ScenarioFilter(*get_filter_parameters(num_scenarios_per_type=100000, limit_total_scenarios=total_scenarios))
    scenarios = builder.get_scenarios(scenario_filter, worker)
    del worker, builder, scenario_filter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(scenarios, device)
    processor.work(save_path, debug=False)

if __name__ == "__main__":
  # 使用 argparse 获取命令行参数
    parser = argparse.ArgumentParser(description="Process data and save the processed results")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the data directory")
    parser.add_argument('--save_path', type=str, required=True, help="Path to the save directory")
    parser.add_argument('--total_scenarios', type=int, default=100000, help="Total number of scenarios to process")
    parser.add_argument('--map_version', type=str, default="nuplan-maps-v1.0", help="Map version")

    # 解析命令行参数
    args = parser.parse_args()

    # 传入命令行参数
    main(args.data_path, args.save_path, total_scenarios=args.total_scenarios, map_version=args.map_version)
