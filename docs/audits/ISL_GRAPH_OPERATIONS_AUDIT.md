# ISL Graph Operations Audit

**Date:** 2026-01-21
**Auditor:** Claude Code
**Scope:** Verify ISL computational purity (no graph-modifying operations)

---

## Executive Summary

| Verdict | Details |
|---------|---------|
| **PARTIAL COMPLIANCE** | ISL performs ONE internal graph transformation |

ISL contains a single graph-modifying operation (`filter_inference_graph`) that removes non-inference nodes internally. **Critically**, this filtered graph is never returned to callers — ISL only returns computational results. However, this operation technically violates the "computational authority only" principle.

---

## Findings

### Finding #1: `filter_inference_graph` (INTERNAL ONLY)

**Location:** [robustness_analyzer_v2.py:56-88](../src/services/robustness_analyzer_v2.py#L56-L88)

**What it does:**
```python
NON_INFERENCE_KINDS = {"decision", "option", "constraint"}

def filter_inference_graph(graph: GraphV2) -> GraphV2:
    """Filter out non-inference nodes and incident edges as a safety net."""
    filtered_nodes = [
        node for node in graph.nodes if node.kind.lower() not in NON_INFERENCE_KINDS
    ]
    # ... also removes edges incident to filtered nodes
    return GraphV2(nodes=filtered_nodes, edges=filtered_edges)
```

**Called at:** [robustness_analyzer_v2.py:514-516](../src/services/robustness_analyzer_v2.py#L514-L516)
```python
filtered_graph = filter_inference_graph(request.graph)
if filtered_graph is not request.graph:
    request = request.model_copy(update={"graph": filtered_graph})
```

**Impact:**
- Removes nodes of kind: `decision`, `option`, `constraint`
- Removes edges connected to those nodes
- Logs warning with removed node/edge counts

**Mitigating factor:** The filtered graph is used ONLY for internal computation. ISL response models (`RobustnessResponseV2`, `ISLResponseV2`) do NOT include the graph — they only return:
- Option results (outcomes, win probabilities)
- Sensitivity analysis
- Robustness metrics
- Metadata (node/edge counts for debugging)

---

## Verification Results

### Node Array Modifications
| Pattern | Found | Context |
|---------|-------|---------|
| `.nodes =` | Yes | Internal variable assignment only |
| `nodes.append` | No | — |
| `nodes.remove` | No | — |

### Edge Array Modifications
| Pattern | Found | Context |
|---------|-------|---------|
| `.edges =` | Yes | Internal variable assignment only |
| `edges.append` | No | — |
| `edges.remove` | No | — |

### Graph Construction
| Pattern | Found | Context |
|---------|-------|---------|
| `GraphV2(...)` | Yes | **ONLY** in `filter_inference_graph` for internal computation |
| `graph.copy()` | Yes | For temporary computation in validators (not returned) |

### ID Modifications
| Pattern | Found | Context |
|---------|-------|---------|
| `node.id =` | No | — |
| `edge.from_ =` | No | — |
| `edge.to =` | No | — |

### Response Verification
| Response Model | Contains Graph? |
|----------------|-----------------|
| `RobustnessResponseV2` | No |
| `ISLResponseV2` | No (only counts) |

---

## Files Examined

| File | Graph Operations? | Notes |
|------|-------------------|-------|
| `robustness_analyzer_v2.py` | **YES** | `filter_inference_graph` — internal only |
| `robustness_analyzer.py` | No | Perturbation on copies for computation |
| `decision_robustness_analyzer.py` | No | Perturbation on copies for computation |
| `identifiability_analyzer.py` | No | Converts to NetworkX for analysis (internal) |
| `causal_validator.py` | No | Creates temporary copies for validation |
| `sequential_decision.py` | No | `_perturb_parameter` on deepcopy (internal) |
| `sensitivity_analyzer.py` | No | Computation only |
| `coherence_analyzer.py` | No | Perturbation on values (not structure) |

---

## Recommendation

### Option A: Accept as Defensive Programming (RECOMMENDED)

**Rationale:**
1. The filter is documented as a "safety net" for malformed input
2. Filtered graph is NEVER returned to callers
3. Schema v2.6 explicitly states these nodes "do not participate in inference"
4. PLoT *should* filter these before sending, but ISL defends against bugs

**Action:** Document the behavior; no code change needed.

### Option B: Migrate to PLoT (Strict Compliance)

If strict SSOT compliance is required:
1. Remove `filter_inference_graph` from ISL
2. Ensure PLoT filters non-inference nodes BEFORE calling ISL
3. ISL should REJECT (422) if non-inference nodes are present

**Risk:** If PLoT has a bug, ISL would include non-inference nodes in computation, producing incorrect results.

### Option C: Fail-Fast Instead of Filter

Change ISL to reject requests containing non-inference nodes:
```python
def validate_inference_graph(graph: GraphV2) -> None:
    non_inference = [n for n in graph.nodes if n.kind.lower() in NON_INFERENCE_KINDS]
    if non_inference:
        raise ValueError(f"Non-inference nodes must be filtered by PLoT: {non_inference}")
```

**Risk:** Breaking change if PLoT isn't updated simultaneously.

---

## Invariant Tests (Recommended)

```python
# tests/invariants/test_isl_purity.py

class TestISLComputationalPurity:
    """Verify ISL doesn't modify graphs in responses."""

    def test_response_does_not_contain_graph(self):
        """ISL responses should not include graph structures."""
        response = RobustnessResponseV2(...)
        assert not hasattr(response, 'graph')
        assert not hasattr(response, 'nodes')
        assert not hasattr(response, 'edges')

    def test_node_ids_in_results_match_input(self):
        """All node IDs in results must exist in input graph."""
        input_graph = create_test_graph()
        input_node_ids = {n.id for n in input_graph.nodes}

        response = analyzer.analyze(request_with_graph(input_graph))

        # Check sensitivity results reference valid nodes
        for sens in response.sensitivity:
            assert sens.from_id in input_node_ids
            assert sens.to_id in input_node_ids

    def test_option_ids_preserved(self):
        """Option IDs in results must match input options."""
        options = [Option(id="opt_a", ...), Option(id="opt_b", ...)]
        response = analyzer.analyze(request_with_options(options))

        result_ids = {r.option_id for r in response.results}
        input_ids = {o.id for o in options}
        assert result_ids == input_ids
```

---

## Conclusion

ISL is **mostly compliant** with computational purity. The single graph-modifying operation (`filter_inference_graph`) is:
- Internal only (never returned)
- Defensive (protects against PLoT bugs)
- Well-documented and logged

**Recommendation:** Accept current behavior as defensive programming. Document in SSOT that ISL may internally filter non-inference nodes as a safety net.
