//===-- StageGraph.h --------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler MLIR Dialects project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGRAPH_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGRAPH_H_

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/STLExtras.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// Iterator over all immediate edges to consumers of a StageOp.
struct ConsumerEdgeIterator
    : llvm::iterator_facade_base<ConsumerEdgeIterator,
                                 std::forward_iterator_tag, OpOperand,
                                 std::ptrdiff_t> {
  [[nodiscard]] static auto begin(StageOp stage) -> ConsumerEdgeIterator {
    return ConsumerEdgeIterator(stage);
  }
  [[nodiscard]] static auto end(StageOp stage) -> ConsumerEdgeIterator {
    return ConsumerEdgeIterator(stage.getDependsInMutable().end());
  }

  explicit ConsumerEdgeIterator() = default;
  explicit ConsumerEdgeIterator(StageOp stage);

  explicit operator bool() const { return use_.getOperand() != nullptr; }

  auto operator++() -> ConsumerEdgeIterator&;

  auto operator*() const -> OpOperand& {
    assert(use_.getOperand());
    return *use_.getOperand();
  }

  [[nodiscard]] auto operator==(const ConsumerEdgeIterator& rhs) const -> bool {
    return operand_ == rhs.operand_ && use_ == rhs.use_;
  }

 private:
  explicit ConsumerEdgeIterator(mlir::OpOperand* operand) : operand_(operand) {}

  mlir::OpOperand* operand_ = nullptr;
  mlir::Value::use_iterator use_;
};

/// Iterator over all immediate consumers of a StageOp.
struct ConsumerIterator
    : llvm::mapped_iterator_base<ConsumerIterator, ConsumerEdgeIterator,
                                 StageOp> {
  [[nodiscard]] static auto begin(StageOp stage) -> ConsumerIterator {
    return ConsumerEdgeIterator::begin(stage);
  }
  [[nodiscard]] static auto end(StageOp stage) -> ConsumerIterator {
    return ConsumerEdgeIterator::end(stage);
  }

  using mapped_iterator_base::mapped_iterator_base;

  [[nodiscard]] auto mapElement(OpOperand& edge) const -> StageOp {
    return cast<StageOp>(edge.getOwner());
  }
};

/// Iterator over all immediate edges from producers of a StageOp.
struct ProducerEdgeIterator
    : llvm::iterator_facade_base<ProducerEdgeIterator,
                                 std::forward_iterator_tag, OpOperand,
                                 std::ptrdiff_t> {
  [[nodiscard]] static auto begin(StageOp stage) -> ProducerEdgeIterator {
    return ProducerEdgeIterator(stage);
  }
  [[nodiscard]] static auto end(StageOp stage) -> ProducerEdgeIterator {
    return ProducerEdgeIterator(stage.getDependsInMutable().end());
  }

  explicit ProducerEdgeIterator() = default;
  explicit ProducerEdgeIterator(StageOp stage);

  explicit operator bool() const { return use_.getOperand() != nullptr; }

  auto operator++() -> ProducerEdgeIterator&;

  auto operator*() const -> OpOperand& {
    assert(use_.getOperand());
    return *use_.getOperand();
  }

  [[nodiscard]] auto operator==(const ProducerEdgeIterator& rhs) const -> bool {
    return operand_ == rhs.operand_ && use_ == rhs.use_;
  }

 private:
  explicit ProducerEdgeIterator(mlir::OpOperand* operand) : operand_(operand) {}

  mlir::OpOperand* operand_ = nullptr;
  mlir::Value::use_iterator use_;
};

/// Iterator over all immediate producers of a StageOp.
struct ProducerIterator
    : llvm::mapped_iterator_base<ProducerIterator, ProducerEdgeIterator,
                                 StageOp> {
  [[nodiscard]] static auto begin(StageOp stage) -> ProducerIterator {
    return ProducerEdgeIterator::begin(stage);
  }
  [[nodiscard]] static auto end(StageOp stage) -> ProducerIterator {
    return ProducerEdgeIterator::end(stage);
  }

  using mapped_iterator_base::mapped_iterator_base;

  [[nodiscard]] auto mapElement(OpOperand& edge) const -> StageOp {
    return cast<StageOp>(edge.getOwner());
  }
};

/// Wrapper around a PipelineOp that exposes the stage dependency graph.
class StageGraph {
 public:
  /// Wrapper around a StageOp that exposes its dependencies.
  struct Node {
    /*implicit*/ Node(StageOp stage) : stage_(stage) {}
    /*implicit*/ operator StageOp() const { return stage_; }

    /// Gets the immediate consumers of this stage.
    [[nodiscard]] auto getConsumers() const
        -> llvm::iterator_range<ConsumerIterator> {
      return {ConsumerIterator::begin(*this), ConsumerIterator::end(*this)};
    }
    /// Gets the immediate producers of this stage.
    [[nodiscard]] auto getProducers() const
        -> llvm::iterator_range<ProducerIterator> {
      return {ProducerIterator::begin(*this), ProducerIterator::end(*this)};
    }

    /// Determines whether this stage (transitively) depends on @p producer .
    [[nodiscard]] auto dependsOn(StageOp producer) const -> bool;

   private:
    StageOp stage_;
  };

  using iterator = Block::op_iterator<StageOp>;

  /*implicit*/ StageGraph(PipelineOp pipeline) : pipeline_(pipeline) {}

  [[nodiscard]] auto begin() const -> iterator {
    return PipelineOp(pipeline_).getBody()->getOps<StageOp>().begin();
  }
  [[nodiscard]] auto end() const -> iterator {
    return PipelineOp(pipeline_).getBody()->getOps<StageOp>().end();
  }

 private:
  PipelineOp pipeline_;
};

}  // namespace mlir::ktdf

template <>
struct llvm::GraphTraits<mlir::ktdf::StageGraph::Node> {
  using GraphType = mlir::ktdf::StageGraph::Node;
  using NodeRef = mlir::ktdf::StageOp;

  [[nodiscard]] static auto getEntryNode(const GraphType& graph) -> NodeRef {
    return graph;
  }

  using EdgeRef = mlir::OpOperand*;
  using ChildEdgeIteratorType = mlir::ktdf::ConsumerEdgeIterator;

  [[nodiscard]] static auto child_edge_begin(NodeRef node)
      -> ChildEdgeIteratorType {
    return ChildEdgeIteratorType::begin(node);
  }
  [[nodiscard]] static auto child_edge_end(NodeRef node)
      -> ChildEdgeIteratorType {
    return ChildEdgeIteratorType::end(node);
  }
  [[nodiscard]] static auto edge_dest(EdgeRef edge) -> NodeRef {
    return llvm::cast<NodeRef>(edge->getOwner());
  }

  using ChildIteratorType = mlir::ktdf::ConsumerIterator;

  [[nodiscard]] static auto child_begin(NodeRef node) -> ChildIteratorType {
    return child_edge_begin(node);
  }
  [[nodiscard]] static auto child_end(NodeRef node) -> ChildIteratorType {
    return child_edge_end(node);
  }
};

template <>
struct llvm::GraphTraits<llvm::Inverse<mlir::ktdf::StageGraph::Node>> {
  using GraphType = Inverse<mlir::ktdf::StageGraph::Node>;
  using NodeRef = mlir::ktdf::StageOp;

  [[nodiscard]] static auto getEntryNode(const GraphType& graph) -> NodeRef {
    return graph.Graph;
  }

  using EdgeRef = mlir::OpOperand*;
  using ChildEdgeIteratorType = mlir::ktdf::ProducerEdgeIterator;

  [[nodiscard]] static auto child_edge_begin(NodeRef node)
      -> ChildEdgeIteratorType {
    return ChildEdgeIteratorType::begin(node);
  }
  [[nodiscard]] static auto child_edge_end(NodeRef node)
      -> ChildEdgeIteratorType {
    return ChildEdgeIteratorType::end(node);
  }
  [[nodiscard]] static auto edge_dest(EdgeRef edge) -> NodeRef {
    return llvm::cast<NodeRef>(edge->getOwner());
  }

  using ChildIteratorType = mlir::ktdf::ProducerIterator;

  [[nodiscard]] static auto child_begin(NodeRef node) -> ChildIteratorType {
    return child_edge_begin(node);
  }
  [[nodiscard]] static auto child_end(NodeRef node) -> ChildIteratorType {
    return child_edge_end(node);
  }
};

template <>
struct llvm::GraphTraits<mlir::ktdf::StageGraph>
    : llvm::GraphTraits<mlir::ktdf::StageGraph::Node> {
  using GraphType = mlir::ktdf::StageGraph;

  using nodes_iterator = GraphType::iterator;
  [[nodiscard]] static auto nodes_begin(GraphType* graph) -> nodes_iterator {
    return graph->begin();
  }
  [[nodiscard]] static auto nodes_end(GraphType* graph) -> nodes_iterator {
    return graph->end();
  }

  static auto getEntryNode(const GraphType&) = delete;
};

template <>
struct llvm::GraphTraits<llvm::Inverse<mlir::ktdf::StageGraph>>
    : GraphTraits<Inverse<mlir::ktdf::StageGraph::Node>> {
  using GraphType = Inverse<mlir::ktdf::StageGraph>;

  using nodes_iterator = mlir::ktdf::StageGraph::iterator;
  [[nodiscard]] static auto nodes_begin(GraphType* graph) -> nodes_iterator {
    return graph->Graph.begin();
  }
  [[nodiscard]] static auto nodes_end(GraphType* graph) -> nodes_iterator {
    return graph->Graph.end();
  }

  static auto getEntryNode(const GraphType&) = delete;
};

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGRAPH_H_
