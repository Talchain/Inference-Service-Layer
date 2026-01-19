# V1 Response Format Deprecation Timeline

## Overview

The V1 response format is maintained for backward compatibility with existing integrations. This document outlines the deprecation timeline and migration path.

---

## Current State

| Client | Response Version | Status |
|--------|------------------|--------|
| PLoT | V2 (enhanced) | Migrated |
| CEE | V1 (legacy) | Pending migration |
| External | Unknown | Audit required |

---

## Timeline

### Phase 1: Soft Deprecation (Current)

**Duration:** Q1 2026

**Actions:**
- V1 remains default for backward compatibility
- V2 available via `?response_version=2` query parameter
- Documentation updated to recommend V2
- Logging added to track V1 vs V2 usage

**Metrics to Track:**
- Request count by `response_version`
- Unique clients using V1
- Error rates by version

### Phase 2: Default Switch

**Target:** Q2 2026

**Criteria:**
- All known clients migrated to V2
- V1 usage < 10% of total requests
- No critical bugs in V2 format

**Actions:**
- Change default from V1 to V2
- Add deprecation warning header for V1 requests
- Update all documentation

### Phase 3: Hard Deprecation

**Target:** Q3 2026

**Criteria:**
- V1 usage < 1% of total requests
- Deprecation warning active for 1+ month
- Communication sent to known V1 users

**Actions:**
- V1 returns 410 Gone with migration instructions
- V1 code paths removed from codebase
- Release as major version (v3.0)

---

## Migration Guide

### Field Mapping

| V1 Field | V2 Field | Notes |
|----------|----------|-------|
| `results[]` | `options[]` | Array name change |
| `results[].option_id` | `options[].id` | Field name change |
| `results[].outcome_distribution.mean` | `options[].outcome.mean` | Nested structure |
| `results[].outcome_distribution.ci_lower` | `options[].outcome.p10` | Renamed |
| `results[].outcome_distribution.ci_upper` | `options[].outcome.p90` | Renamed |
| `sensitivity[]` | `robustness.fragile_edges[]` | Restructured |

### New V2-Only Fields

| Field | Description |
|-------|-------------|
| `options[].outcome.validity_ratio` | Fraction of valid MC samples |
| `options[].status` | `computed`, `partial`, or `failed` |
| `robustness.fragile_edges[].alternative_winner_id` | Option winning when edge weak |
| `robustness.fragile_edges[].switch_probability` | P(alternative wins) |

### Example Migration

**V1 Request:**
```bash
curl -X POST ".../analyze/v2"  # Default is V1
```

**V2 Request:**
```bash
curl -X POST ".../analyze/v2?response_version=2"
```

---

## Client Communication Template

```
Subject: ISL V1 Response Format Deprecation Notice

The V1 response format for ISL robustness analysis will be deprecated
according to the following timeline:

- Q2 2026: Default changes to V2 (V1 still available)
- Q3 2026: V1 removed entirely

Action Required:
1. Test your integration with ?response_version=2
2. Update your response parsing to handle V2 format
3. Confirm migration complete by [date]

Migration guide: [link to this document]
```

---

## Rollback Plan

If critical issues arise after Phase 2:
1. Revert default to V1 via feature flag
2. Extend Phase 2 timeline
3. Address issues in V2 format
4. Re-attempt default switch

---

## Success Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|----------------|----------------|----------------|
| V2 adoption | 50% | 90% | 99%+ |
| V2 error rate | < V1 rate | < V1 rate | N/A |
| Migration tickets | Documented | Resolved | N/A |
