from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import numpy.typing as npt

Transform = npt.NDArray[np.float32]  # 4x4 homogeneous transformation matrix

PointCloud = npt.NDArray[np.float32]  # Nx4 array of lidar points (TODO: wrap in dataclass)

VectorLayer = gpd.GeoDataFrame


class SemanticMapLayer(IntEnum):
    """
    Enum for SemanticMapLayers.
    """

    LANE = 0  #普通车道，表示可通行的标准车道（包含方向、边界等信息）。
    INTERSECTION = 1 #交叉口区域，包含多个 LANE 和 LANE_CONNECTOR 交汇的地方。
    STOP_LINE = 2 #停车线，车辆在红灯或 STOP SIGN 处必须停下的线。
    TURN_STOP = 3 #	转向停止点，比如左转或右转等待区的停止位置。
    CROSSWALK = 4 # 人行横道，行人穿越马路的区域。
    DRIVABLE_AREA = 5 #	可行驶区域，表示汽车可以行驶的路面（包括无标线的区域）。
    YIELD = 6  #让行标志，车辆在进入某些路段时需要让行。
    TRAFFIC_LIGHT = 7 #交通信号灯，控制通行权（红绿灯等）。
    STOP_SIGN = 8  #STOP 停止标志，要求驾驶员完全停车再继续行驶。
    EXTENDED_PUDO = 9 #扩展的上下车区域，比标准 PUDO 更大，可能包括更多等候区。
    SPEED_BUMP = 10 #减速带，用于强制车辆减速。
    LANE_CONNECTOR = 11  #车道连接器，表示 车道与车道之间的连接，如匝道或转向专用道。
    BASELINE_PATHS = 12 #车道的参考路径，提供一条用于引导车辆通行的中心线（baseline）。
    BOUNDARIES = 13 #道路边界，定义道路的物理边界（例如护栏、路沿）。
    WALKWAYS = 14 #人行道，专门供行人行走的步道。
    CARPARK_AREA = 15 #	停车场区域，包括商业停车场、路边停车区等。
    PUDO = 16  #乘客上下车区域（Pick-up/Drop-off），专门用于乘客上下车的点。
    ROADBLOCK = 17 # 代表的是 一整块道路结构，而不是单独的车道（LANE
    ROADBLOCK_CONNECTOR = 18

    @classmethod
    def deserialize(cls, layer: str) -> SemanticMapLayer:
        """Deserialize the type when loading from a string."""
        return SemanticMapLayer.__members__[layer]


class LaneConnectorType(IntEnum):
    """
    Enum for LaneConnectorType.
    """

    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    UTURN = 3
    UNKNOWN = 4


class StopLineType(IntEnum):
    """
    Enum for StopLineType.
    """

    PED_CROSSING = 0
    STOP_SIGN = 1
    TRAFFIC_LIGHT = 2
    TURN_STOP = 3
    YIELD = 4
    UNKNOWN = 5


class PudoType(IntEnum):
    """
    Enum for PudoType
    """

    PICK_UP_DROP_OFF = 0
    PICK_UP_ONLY = 1
    DROP_OFF_ONLY = 2
    UNKNOWN = 3


class IntersectionType(IntEnum):
    """
    Enum for IntersectionType.
    """

    DEFAULT = 0
    TRAFFIC_LIGHT = 1
    STOP_SIGN = 2
    LANE_BRANCH = 3
    LANE_MERGE = 4
    PASS_THROUGH = 5


class TrafficLightStatusType(IntEnum):
    """
    Enum for TrafficLightStatusType.
    """

    GREEN = 0
    YELLOW = 1
    RED = 2
    UNKNOWN = 3

    def serialize(self) -> str:
        """Serialize the type when saving."""
        return self.name

    @classmethod
    def deserialize(cls, key: str) -> TrafficLightStatusType:
        """Deserialize the type when loading from a string."""
        return TrafficLightStatusType.__members__[key]


@dataclass
class RasterLayer:
    """
    Wrapper dataclass of a layer of the rasterized map.
    """

    data: npt.NDArray[np.uint8]  # raster image as numpy array
    precision: np.float64  # [m] precision of map
    transform: Transform  # transform from physical to pixel coordinates


@dataclass
class VectorMap:
    """
    Dataclass mapping SemanticMapLayers to associated VectorLayer.
    """

    layers: Dict[SemanticMapLayer, VectorLayer]  # type: ignore


@dataclass
class RasterMap:
    """
    Dataclass mapping SemanticMapLayers to associated RasterLayer.
    """

    layers: Dict[SemanticMapLayer, RasterLayer]


@dataclass
class TrafficLightStatusData:
    """Traffic light status."""

    status: TrafficLightStatusType  # Status: green, red
    lane_connector_id: int  # lane connector id, where this traffic light belongs to
    timestamp: int  # Timestamp

    def serialize(self) -> Dict[str, Any]:
        """Serialize traffic light status."""
        return {
            'status': self.status.serialize(),
            'lane_connector_id': self.lane_connector_id,
            'timestamp': self.timestamp,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> TrafficLightStatusData:
        """Deserialize a dict of data to this class."""
        return TrafficLightStatusData(
            status=TrafficLightStatusType.deserialize(data['status']),
            lane_connector_id=data['lane_connector_id'],
            timestamp=data['timestamp'],
        )


@dataclass
class TrafficLightStatuses:
    """
    Collection of TrafficLightStatusData at a time step.
    """

    traffic_lights: List[TrafficLightStatusData]
