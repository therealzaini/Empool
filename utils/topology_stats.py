from dataclasses import dataclass
from typing import List


@dataclass
class TopologyStats:
    num_nodes: List[int]
    num_sinks: List[int]
    sink_ratios: List[float]
    global_sink_ratio: float