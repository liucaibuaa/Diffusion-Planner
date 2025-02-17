import numpy as np

# from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_pool import Task

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
agent_past_process,
sampled_tracked_objects_to_array_list,
sampled_static_objects_to_array_list
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.utils import (convert_to_model_inputs,
get_scenario_map,
get_filter_parameters
)

class DataProcessor(object):
    def __init__(self, config, scenarios):

        self.num_agents = 20 #[int]
        self.num_static = 20 #[int]
        self.max_ped_bike = 10 # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 80 # [m] query radius scope relative to the current pose.

        self._map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'] # name of map features to be extracted.
        self._max_elements = {'LANE': 40, 'LEFT_BOUNDARY': 10, 'RIGHT_BOUNDARY': 10, 'ROUTE_LANES': 10} # maximum number of elements to extract per feature layer.
        self._max_points = {'LANE': 50, 'LEFT_BOUNDARY': 50, 'RIGHT_BOUNDARY': 50, 'ROUTE_LANES': 50} # maximum number of points per feature to extract per feature layer.
        # self._vehicle_parameters = get_pacifica_parameters()

        self._scenarios = scenarios
        self.past_time_horizon = 2 #[seconds]
        self._interpolation_method = 'linear' # Interpolation method to apply when interpolating to maintain fixed size map elements.

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

if __name__ == "__main__":
  map_version = "nuplan-maps-v1.0"
  data_path = "/cailiu2/Diffusion-Planner/data"
  map_path = "/cailiu2/Diffusion-Planner/data/maps"
  save_path = "/cailiu2/Diffusion-Planner/data/processed"
  total_scenarios = 30000
  scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
  builder = NuPlanScenarioBuilder(data_path, map_path, None, None, map_version, scenario_mapping = scenario_mapping)
  worker = SingleMachineParallelExecutor(use_process_pool=True)
  scenario_filter = ScenarioFilter(*get_filter_parameters(num_scenarios_per_type=30000,
                                                            limit_total_scenarios=total_scenarios))
  scenarios = builder.get_scenarios(scenario_filter, worker)
  del worker, builder, scenario_filter
  processor = DataProcessor(scenarios)
  processor.work(save_path, debug=False)


