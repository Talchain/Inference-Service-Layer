# Archived: Habermas Machine Services

## Status
**ARCHIVED** - Deferred to TAE (Team Alignment Engine) PoC v02

## Reason for Archival
During ISL PoC v01 finalization (Phase 4C), we simplified the architecture to focus exclusively on PLoT (Program-Oriented Learning and Teaching) integration. The Habermas Machine deliberation capabilities are being deferred to TAE PoC v02, where they will be the central focus.

## What Was Archived
The following Habermas Machine services were moved to this archive on 2025-11-21:

### Core Orchestration
- `deliberation_orchestrator.py` - Main orchestrator for multi-round deliberation
- `deliberation_factory.py` - Factory for creating deliberation services

### LLM-Powered Components
- `value_extractor_llm.py` - LLM-based value and concern extraction
- `consensus_generator_llm.py` - LLM-based consensus generation
- `common_ground_finder.py` - Common ground identification

### Rule-Based Fallbacks
- `value_extractor.py` - Rule-based value extraction
- `consensus_generator.py` - Rule-based consensus generation

## Architecture Context

### ISL PoC v01 (Current)
**Architecture:** PLoT-only consumer
- PLoT Engine is the SOLE caller of ISL
- CEE (Causal Exploration Environment) and UI do NOT call ISL directly
- All causal operations flow through PLoT → ISL

**Scope:**
- ✅ Causal validation (do-calculus, identifiability)
- ✅ Counterfactual prediction (interventions, what-if analysis)
- ✅ Robustness analysis (sensitivity testing)
- ❌ Habermas Machine deliberation (deferred)

### TAE PoC v02 (Future)
The Habermas Machine will be the central component of the Team Alignment Engine, providing:
- Multi-stakeholder deliberation
- Value extraction and alignment
- Consensus building
- Common ground identification
- Communicative rationality principles

**Integration:** TAE will potentially use ISL for causal reasoning, but will own all deliberation logic.

## Technical Details

### Deliberation Orchestrator
The orchestrator managed multi-round deliberations with:
- Value extraction from position statements
- Common ground identification across perspectives
- Consensus generation using Habermasian principles
- Round-by-round convergence tracking

### LLM Integration
LLM-powered components used:
- OpenAI GPT-4 for consensus generation
- GPT-3.5-turbo for value extraction
- Anthropic Claude as fallback
- Cost tracking (target: <$0.10 per round)
- Redis/memory caching for prompt reuse

### Rule-Based Fallbacks
Keyword-based extraction and template-based consensus provided fallback when:
- LLM APIs unavailable
- Budget limits exceeded
- Session costs too high

## API Endpoints Removed
The following deliberation endpoints were disabled in ISL PoC v01:
- `POST /api/v1/deliberation/deliberate` - Start deliberation
- `POST /api/v1/deliberation/round` - Run deliberation round
- `GET /api/v1/deliberation/status/{session_id}` - Get status
- `POST /api/v1/preferences/elicit` - Elicit user preferences
- (See `src/api/deliberation.py` for full endpoint list)

## Migration Path to TAE

### For TAE PoC v02 Development:
1. **Copy archived files** to TAE repository
2. **Update imports** to reflect TAE package structure
3. **Enhance orchestrator** for team-specific workflows
4. **Add TAE-specific features:**
   - Team role management
   - Multi-session deliberation tracking
   - Decision history and audit trail
   - Value alignment metrics
   - Consensus quality scoring

### Integration with ISL:
TAE may call ISL for causal reasoning within deliberations:
```python
# Example: TAE uses ISL for causal validation during deliberation
async def validate_intervention_proposal(proposal):
    isl_response = await isl_client.validate_causal(
        dag=proposal.causal_model,
        treatment=proposal.intervention,
        outcome=proposal.target_outcome
    )

    if isl_response.status == "identifiable":
        # Continue deliberation with validated causal model
        deliberation.update_context(isl_response)
    else:
        # Flag to team: causal model needs refinement
        deliberation.flag_causal_issue(isl_response.why_not_identifiable)
```

## Dependencies Still in ISL
The following LLM infrastructure remains in ISL (needed for potential TAE integration):
- `src/config/llm_config.py` - LLM configuration
- `src/services/llm_client.py` - LLM client with cost tracking
- `src/infrastructure/memory_cache.py` - In-memory cache (used by LLM client)

These are kept because:
1. Future ISL features may use LLM (e.g., natural language causal queries)
2. TAE can leverage ISL's LLM client if desired
3. Clean separation of concerns (LLM client ≠ Habermas Machine)

## Test Files
Related test files (kept for reference, but will fail until TAE integration):
- `tests/unit/test_deliberation_orchestrator.py`
- `tests/unit/test_value_extractor.py`
- `tests/unit/test_consensus_generator.py`
- `tests/integration/test_deliberation_workflow.py`

## Documentation References
Related documentation (now outdated for ISL):
- Phase 3 Habermas Implementation docs (archived)
- Workshop materials mentioning deliberation (to be updated)

## Questions?
Contact:
- **ISL Team:** [ISL team contact]
- **TAE Team:** [TAE team contact]
- **Architecture Decisions:** See ISL Phase 4C Development Brief

## Restoration (if needed)
To restore these services to ISL (unlikely):
```bash
# Copy files back to src/services/
cp src/services/_archived/habermas/*.py src/services/

# Re-enable deliberation endpoints in src/api/main.py
# Uncomment: app.include_router(deliberation_router, tags=["deliberation"])

# Restore tests
# Update imports in test files
# Run: pytest tests/unit/test_deliberation_*

# Update documentation
# Restore deliberation sections in API_QUICK_REFERENCE.md
```

## Timeline
- **Archived:** 2025-11-21 (ISL Phase 4C)
- **Target Restoration:** TAE PoC v02 (Q1 2026, tentative)
- **Status:** Clean separation complete, ready for TAE development

---

**This archival ensures clean separation between ISL (causal reasoning) and TAE (team alignment), while preserving all Habermas Machine work for future TAE development.**
