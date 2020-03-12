# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import networkx as nx
import numpy as np
import math

from tensorflow.python.ipu import sharding
from tensorflow.python.ipu.sharding_utils import assign_shard, \
    convert_inference_ops_to_nx, calculate_memory, tensor_memory_use, \
    children, find_all_subgraphs, group_subgraphs, is_splitting_edge, \
    is_splitting_node, logging_helper
from tensorflow.python.ipu.autoshard_cnn import convert_ops_to_nx
from tensorflow.python.ops.variables import trainable_variables
from tensorflow.python.platform import tf_logging as logging

prohibited_ops = frozenset(["NextIteration", "PopDatastreamInfeedDequeue"])


def find_all_spliting_nodes(input_ts,
                            output_ts,
                            node_matcher=None,
                            frozen_inference=False,
                            ):
  output_op = output_ts.op
  input_op = input_ts.op
  ipu_ops = list(
      filter(lambda o: 'IPU' in o.device, output_op.graph.get_operations()))
  if len(ipu_ops) == 0:
    raise ValueError("No ops placed on IPU device to shard.")

  fwd_ops = []
  marked_collection = output_op.graph.get_collection(sharding._IPU_AUTOSHARD)
  if len(marked_collection) > 0:
    fwd_ops = marked_collection
  else:
    for op in ipu_ops:
      if not any([s in op.name.lower() for s in ['gradients/', '/update_']]):
        fwd_ops.append(op)
  bwd_ops = [o for o in ipu_ops if o not in fwd_ops]
  fwd_ops = [o for o in fwd_ops if o.type not in prohibited_ops]

  if input_op not in fwd_ops:
    input_op = [op for op in input_ts.consumers() if op in fwd_ops][0]

  if frozen_inference:
    graph = convert_inference_ops_to_nx(fwd_ops)
  else:
    graph = convert_ops_to_nx(fwd_ops, bwd_ops)

  # Check graph is a single weakly connected component
  # if not find the component with the output op in and use that
  weakly_connected_components = list(nx.weakly_connected_components(graph))
  graph_fwd = None
  for g in weakly_connected_components:
    if output_op.name in g:
      graph_fwd = graph.subgraph(g)
      break
  fwd_ops = [op for op in fwd_ops if op.name in graph_fwd.nodes]

  if nx.number_weakly_connected_components(graph_fwd) != 1:
    raise RuntimeError(
        "Error: number of disconnected subgraphs in auto-sharder is {}".format(
            nx.number_weakly_connected_components(graph)))
  splitting_nodes = []
  for node in graph_fwd.nodes:
    if is_splitting_node(graph_fwd, node, input_op.name, output_op.name):
      splitting_nodes.append(node)
  
  if node_matcher and callable(node_matcher):
    splitting_nodes = list(
        filter(lambda e: node_matcher(e), splitting_nodes))

  return splitting_nodes


def manual_sharding(input_ts,
                    output_ts,
                    splitting_nodes,
                    frozen_inference=False):
  if not isinstance(splitting_nodes, (tuple, list)) or len(splitting_nodes) == 0:
    raise TypeError("Invalid splitting_nodes.")

  output_op = output_ts.op
  input_op = input_ts.op
  ipu_ops = list(
      filter(lambda o: 'IPU' in o.device, output_op.graph.get_operations()))
  if len(ipu_ops) == 0:
    raise ValueError("No ops placed on IPU device to shard.")

  fwd_ops = []
  marked_collection = output_op.graph.get_collection(sharding._IPU_AUTOSHARD)
  if len(marked_collection) > 0:
    fwd_ops = marked_collection
  else:
    for op in ipu_ops:
      if not any([s in op.name.lower() for s in ['gradients/', '/update_']]):
        fwd_ops.append(op)
  bwd_ops = [o for o in ipu_ops if o not in fwd_ops]
  fwd_ops = [o for o in fwd_ops if o.type not in prohibited_ops]

  if input_op not in fwd_ops:
    input_op = [op for op in input_ts.consumers() if op in fwd_ops][0]

  if frozen_inference:
    graph = convert_inference_ops_to_nx(fwd_ops)
  else:
    graph = convert_ops_to_nx(fwd_ops, bwd_ops)
  
  # Check graph is a single weakly connected component
  # if not find the component with the output op in and use that
  weakly_connected_components = list(nx.weakly_connected_components(graph))
  graph_fwd = None
  for g in weakly_connected_components:
    if output_op.name in g:
      graph_fwd = graph.subgraph(g)
      break
  fwd_ops = [op for op in fwd_ops if op.name in graph_fwd.nodes]

  if nx.number_weakly_connected_components(graph_fwd) != 1:
    raise RuntimeError(
        "Error: number of disconnected subgraphs in auto-sharder is {}".format(
            nx.number_weakly_connected_components(graph)))
  
  splitting_edges = []
  for node in splitting_nodes:
    if not isinstance(node, str):
      raise TypeError("splitting_edges format error!")
    if node not in graph_fwd.nodes:
      raise ValueError("node {} does not exist in graph_fwd. valid node:{}".format(node, graph_fwd.nodes))
    if not is_splitting_node(graph_fwd, node, input_op.name, output_op.name):
      raise ValueError("node {} is not a splitting node.")
    splitting_edges.append([(node, v) for v in graph_fwd.successors(node)])
  logging.debug('Splitting nodes ' + str(splitting_nodes))

  # Given the splitting edges found find all of the subgraphs created and order them
  sub_graphs = find_all_subgraphs(graph_fwd, splitting_edges, input_op.name,
                                  output_op.name, [op.name for op in fwd_ops])
  assign_shard(fwd_ops, ipu_ops, sub_graphs)