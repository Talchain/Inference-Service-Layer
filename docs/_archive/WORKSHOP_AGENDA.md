# ISL Integration Workshop

## Audience
- **PLoT Engine team** - Causal model validation integration
- **CEE team** - Assumption transparency and sensitivity analysis
- **UI/Scenario Sandbox team** - Preferences and deliberation features

## Duration
2 hours

## Pre-Workshop Setup (Send 1 Week Before)

### Access Credentials
- **ISL Staging URL:** `https://isl-staging.onrender.com`
- **API Keys:** Individual keys per team (sent separately)
- **Grafana Dashboard:** `https://grafana.olumi.com/d/isl-overview`
- **Documentation:** `https://docs.olumi.com/isl`

### Materials to Review
1. **Technical Specification:** `docs/TECHNICAL_SPECIFICATIONS.md`
2. **API Quick Reference:** `docs/API_QUICK_REFERENCE.md`
3. **TypeScript Client README:** `clients/typescript/README.md`

### Setup Instructions
```bash
# Install TypeScript client
npm install @olumi/isl-client

# Or via Yarn
yarn add @olumi/isl-client

# Verify installation
npm list @olumi/isl-client
```

### Sandbox Environment
- Each team gets isolated namespace
- Pre-populated test data
- Dedicated monitoring dashboard per team
- Rate limits: 100 req/min (staging)

---

## Agenda

### Part 1: ISL Overview (20 min)
**Presenter:** Paul

#### ISL Mission & Capabilities
- **Why ISL exists:** Fill critical gap in ML-driven decision support
- **Research foundations:**
  - **Y₀:** State-of-the-art causal identification (Columbia/Judea Pearl)
  - **ActiVA:** Efficient Bayesian preference learning
  - **FACET:** Region-based robustness analysis
  - **Habermas Machine:** Democratic deliberation with LLM enhancement

#### Architecture Overview
```
┌──────────┐     validate     ┌─────────┐     analyze      ┌──────────┐
│   PLoT   │───────────────>│   ISL   │─────────────────>│  CEE UI  │
│  Engine  │<───────────────│  API    │<─────────────────│          │
└──────────┘     results     └─────────┘     transparency  └──────────┘
                                    │
                                    │ deliberate
                                    ▼
                            ┌─────────────────┐
                            │  Scenario       │
                            │  Sandbox UI     │
                            └─────────────────┘
```

#### Performance Characteristics
- **Latency:** 100-300ms (validation), 1-5s (robustness), 1-3s (deliberation)
- **Throughput:** 50 concurrent users validated in load testing
- **Availability:** 99.5% SLO target
- **Error Rate:** <1% target

#### Cost Model (LLM Usage)
- **Deliberation (LLM-powered):** ~$0.08 per round
- **Value Extraction:** ~$0.02 per position
- **Consensus Generation:** ~$0.06 per consensus statement
- **Daily Budget:** $100 (pilot limit)
- **Fallback:** Automatic switch to rule-based if budget exceeded

---

### Part 2: API Deep Dive (30 min)
**Presenter:** Technical Lead

#### 1. Causal Validation (`POST /api/v1/causal/validate`)

**Live Demo:**
```typescript
import { ISLClient } from '@olumi/isl-client';

const isl = new ISLClient('https://isl-staging.onrender.com', API_KEY);

const validation = await isl.validateCausal({
  dag: {
    nodes: ["price", "demand", "revenue"],
    edges: [["price", "demand"], ["demand", "revenue"]]
  },
  treatment: "price",
  outcome: "revenue"
});

console.log(validation);
// {
//   status: "identifiable",
//   method: "backdoor",
//   formula: "P(revenue|do(price)) = Σ_demand P(revenue|price,demand)P(demand)",
//   required_assumptions: [...],
//   explanation: {...}
// }
```

**Response Structure Walkthrough:**
- `status`: "identifiable", "not_identifiable", "conditional"
- `method`: "backdoor", "frontdoor", "instrumental", etc.
- `formula`: Mathematical identification formula
- `required_assumptions`: What must be true for this to work
- `adjustment_set`: Which variables to control for
- `explanation`: Human-readable why/why not

**Error Handling Examples:**
```typescript
try {
  const result = await isl.validateCausal({...});
} catch (error) {
  if (error.status === 400) {
    // Invalid DAG structure
    console.error('DAG validation failed:', error.detail);
  } else if (error.status === 429) {
    // Rate limited
    console.error('Too many requests, retry after:', error.retryAfter);
  }
}
```

#### 2. Counterfactual Generation (`POST /api/v1/causal/counterfactual`)

**Live Demo:**
```typescript
const counterfactual = await isl.generateCounterfactual({
  causal_model: {
    nodes: ["price", "demand", "revenue"],
    edges: [["price", "demand"], ["demand", "revenue"]],
    structural_equations: {
      demand: "1000 - 10 * price + noise",
      revenue: "price * demand"
    }
  },
  intervention: { price: 55 },
  outcome_variables: ["revenue"],
  samples: 1000
});

console.log(counterfactual.prediction);
// {
//   point_estimate: { revenue: 105000 },
//   confidence_interval: {
//     revenue: { p10: 95000, p50: 105000, p90: 115000 }
//   }
// }
```

**Intervention Specification:**
- Single variable: `{ price: 55 }`
- Multiple variables: `{ price: 55, marketing_spend: 10000 }`
- Supports continuous and discrete variables

**Uncertainty Quantification:**
- Monte Carlo sampling (default: 1000 samples)
- Confidence intervals at p10, p50, p90
- Sensitivity ranges (optimistic/pessimistic)

#### 3. Robustness Analysis (`POST /api/v1/robustness/analyze`)

**Live Demo:**
```typescript
const robustness = await isl.analyzeRobustness({
  causal_model: {...},
  intervention_proposal: { price: 55 },
  target_outcome: { revenue: [95000, 105000] },
  perturbation_radius: 0.1,  // ±10%
  min_samples: 100
});

console.log(robustness.analysis);
// {
//   status: "robust",
//   robustness_score: 0.75,
//   robust_regions: [
//     { variable_ranges: { price: [52.0, 58.0] } }
//   ],
//   interpretation: "ROBUST RECOMMENDATION...",
//   recommendation: "Proceed with confidence..."
// }
```

**Fragility Warnings Interpretation:**
- `status: "fragile"` → Small changes break recommendation
- `status: "robust"` → Wide operating range available
- `robustness_score: 0-1` → Higher = more robust
- `robust_regions` → Safe intervention ranges

**Operating Ranges:**
- Tell user: "Any price between 52-58 works"
- vs. Fragile: "Only price 54.8-55.2 works"

#### 4. Preference Learning (`POST /api/v1/preferences/elicit`)

**Live Demo:**
```typescript
const preferences = await isl.elicitPreferences({
  user_id: "user_123",
  causal_model: {...},
  context: { revenue: 100000, churn: 0.10 },
  feature_names: ["revenue", "churn"]
});

if (preferences.status === "eliciting") {
  // Show counterfactual question
  const { scenario_a, scenario_b } = preferences.scenario_pair;
  console.log("Which do you prefer?");
  console.log("A:", scenario_a);  // { revenue: 110000, churn: 0.12 }
  console.log("B:", scenario_b);  // { revenue: 105000, churn: 0.08 }

  // User responds
  const updated = await isl.submitPreference({
    user_id: "user_123",
    preference_id: preferences.preference_id,
    choice: "B"
  });
} else if (preferences.status === "converged") {
  // Preferences learned!
  console.log("Value function:", preferences.value_function);
  // { weights: { revenue: 0.7, churn: 0.3 }, confidence: 0.85 }
}
```

**Bayesian Updates:**
- Starts with uninformative prior
- Each answer refines posterior
- Typically converges in 5-10 queries
- Convergence detection automatic

**Convergence Detection:**
- Posterior entropy < threshold
- Information gain < threshold
- Max queries reached (failsafe)

#### 5. Team Deliberation (`POST /api/v1/deliberation/deliberate`)

**Live Demo:**
```typescript
const deliberation = await isl.conductDeliberation({
  decision_context: "Should we increase price to £55?",
  positions: [
    {
      member_id: "pm_001",
      position_statement: "I support this because revenue increases and risk is low.",
      timestamp: new Date().toISOString()
    },
    {
      member_id: "eng_001",
      position_statement: "I'm concerned about implementation complexity.",
      timestamp: new Date().toISOString()
    }
  ]
});

console.log(deliberation);
// {
//   session_id: "delib_abc123",
//   round_number: 1,
//   common_ground: {
//     shared_values: [
//       { value_name: "revenue_growth", agreement_score: 0.9, ... }
//     ],
//     agreement_level: 0.75
//   },
//   consensus_statement: {
//     text: "We agree that revenue growth is important...",
//     support_score: 0.8
//   },
//   status: "active"
// }
```

**Value Extraction from Positions:**
- LLM extracts: values, concerns, rationales
- Rule-based fallback if LLM unavailable
- Identifies: "revenue_growth", "implementation_complexity", etc.

**Consensus Generation:**
- Synthesizes common ground
- Acknowledges concerns
- Proposes integrated solution
- Natural language (not templates)

---

### Part 3: TypeScript Client (20 min)
**Presenter:** Frontend Lead

#### Installation & Setup
```bash
npm install @olumi/isl-client
```

```typescript
// src/lib/isl.ts
import { ISLClient } from '@olumi/isl-client';

export const islClient = new ISLClient(
  process.env.NEXT_PUBLIC_ISL_URL || 'https://isl-staging.onrender.com',
  process.env.ISL_API_KEY!
);
```

#### Type-Safe API Calls
```typescript
// All methods fully typed!
const validation: CausalValidationResponse = await islClient.validateCausal({
  dag: { nodes: [...], edges: [...] },
  treatment: "X",
  outcome: "Y"
});

// TypeScript catches errors at compile time
validation.status;  // ✓ Type: "identifiable" | "not_identifiable" | "conditional"
validation.invalid;  // ✗ Compile error: Property 'invalid' does not exist
```

#### React Hooks Usage
```typescript
import { useCausalValidation } from '@olumi/isl-client/hooks';

function CausalValidator() {
  const { validate, loading, error, result } = useCausalValidation(islClient);

  const handleValidate = async () => {
    await validate({
      dag: plotModel.dag,
      treatment: plotModel.intervention,
      outcome: plotModel.outcome
    });
  };

  return (
    <div>
      <button onClick={handleValidate} disabled={loading}>
        Validate
      </button>

      {loading && <Spinner />}
      {error && <ErrorDisplay error={error} />}
      {result && <ValidationResult data={result} />}
    </div>
  );
}
```

#### Error Handling Patterns
```typescript
// Built-in retry with exponential backoff
const client = new ISLClient(url, apiKey, {
  retries: 3,
  retryDelay: 1000,
  timeout: 30000
});

// Automatic error normalization
try {
  await client.validateCausal({...});
} catch (error) {
  // All errors normalized to ISLError type
  console.error(error.code);     // e.g., "RATE_LIMIT_EXCEEDED"
  console.error(error.message);  // Human-readable
  console.error(error.retryAfter);  // Available for 429 errors
}
```

#### Example: Build Simple Deliberation UI
```typescript
import { useDeliberation } from '@olumi/isl-client/hooks';

function DeliberationSession({ decisionContext }: Props) {
  const { deliberate, loading, session } = useDeliberation(islClient);
  const [positions, setPositions] = useState<Position[]>([]);

  const handleSubmitPositions = async () => {
    await deliberate({
      decision_context: decisionContext,
      positions: positions.map(p => ({
        member_id: p.memberId,
        position_statement: p.statement,
        timestamp: new Date().toISOString()
      }))
    });
  };

  return (
    <div>
      <h2>Team Deliberation: {decisionContext}</h2>

      <PositionInput
        onAddPosition={(p) => setPositions([...positions, p])}
      />

      <button onClick={handleSubmitPositions} disabled={loading}>
        Start Deliberation
      </button>

      {session && (
        <DeliberationResults
          commonGround={session.common_ground}
          consensus={session.consensus_statement}
          agreementLevel={session.common_ground.agreement_level}
        />
      )}
    </div>
  );
}
```

---

### Part 4: Hands-On Integration (40 min)

**Break into 3 groups (PLoT, CEE, UI)**

#### Group 1: PLoT + ISL
**Task:** Integrate causal validation into PLoT pipeline

**Scenario:** Before executing a PLoT program, validate that the causal effect is identifiable.

```typescript
// src/plot/executor.ts
import { ISLClient } from '@olumi/isl-client';

const isl = new ISLClient(process.env.ISL_URL!, process.env.ISL_API_KEY!);

async function executePLoTProgram(program: PLoTProgram) {
  // Step 1: Validate causal model
  const validation = await isl.validateCausal({
    dag: program.causalGraph,
    treatment: program.intervention,
    outcome: program.outcome
  });

  if (validation.status !== 'identifiable') {
    // Show user why model won't work
    throw new Error(
      `Causal effect not identifiable. ${validation.explanation.why_not_identifiable}`
    );
  }

  // Step 2: Show required assumptions to user
  console.log('Required assumptions:');
  validation.required_assumptions.forEach(a => {
    console.log(`- ${a.description} (evidence: ${a.evidence_strength})`);
  });

  // Step 3: Execute PLoT program with validated model
  const result = await runPLoT(program);
  return result;
}
```

**Exercise:**
1. Create a PLoTProgram with a non-identifiable DAG
2. See validation fail
3. Fix the DAG
4. See validation succeed

#### Group 2: CEE + ISL
**Task:** Add assumption transparency display in CEE UI

**Scenario:** When showing a causal claim, display all assumptions required for it to be valid.

```typescript
// src/cee/AssumptionsList.tsx
import { CausalValidationResponse } from '@olumi/isl-client';

interface Props {
  validation: CausalValidationResponse;
}

function AssumptionsList({ validation }: Props) {
  return (
    <div className="assumptions-panel">
      <h3>Required Assumptions</h3>
      <p className="help-text">
        For this causal claim to be valid, the following must be true:
      </p>

      {validation.required_assumptions.map((assumption, i) => (
        <AssumptionCard
          key={i}
          name={assumption.name}
          description={assumption.description}
          evidence={assumption.evidence_strength}
          consequences={assumption.consequences_if_violated}
          testable={assumption.testable}
        />
      ))}

      {validation.status === 'conditional' && (
        <ConditionalWarning>
          This effect is only identifiable if you control for:{' '}
          {validation.adjustment_set.join(', ')}
        </ConditionalWarning>
      )}
    </div>
  );
}

function AssumptionCard({ name, description, evidence, consequences, testable }: AssumptionProps) {
  const evidenceColor = {
    strong: 'green',
    medium: 'yellow',
    weak: 'red'
  }[evidence];

  return (
    <div className="assumption-card">
      <div className="assumption-header">
        <h4>{name.replace(/_/g, ' ').toUpperCase()}</h4>
        <Badge color={evidenceColor}>{evidence} evidence</Badge>
      </div>

      <p>{description}</p>

      {testable && (
        <InfoBox>
          This assumption can be tested empirically. Recommend robustness check.
        </InfoBox>
      )}

      <details>
        <summary>What if this is violated?</summary>
        <p className="consequences">{consequences}</p>
      </details>
    </div>
  );
}
```

**Exercise:**
1. Create a causal validation with multiple assumptions
2. Display assumptions in UI
3. Color-code by evidence strength
4. Add expandable "consequences" section

#### Group 3: UI + ISL
**Task:** Build preference elicitation component for Scenario Sandbox

**Scenario:** Learn user preferences through counterfactual questions.

```typescript
// src/preferences/PreferenceLearning.tsx
import { usePreferences } from '@olumi/isl-client/hooks';

function PreferenceLearning({ userId, scenario }: Props) {
  const { elicit, submit, loading, query, preferences } = usePreferences(islClient);

  const handleStartElicitation = async () => {
    await elicit({
      user_id: userId,
      causal_model: scenario.causalModel,
      context: scenario.baseline,
      feature_names: ['revenue', 'churn', 'satisfaction']
    });
  };

  const handleAnswer = async (choice: 'A' | 'B') => {
    await submit({
      user_id: userId,
      preference_id: query!.preference_id,
      choice: choice
    });
  };

  if (!query && !preferences) {
    return (
      <button onClick={handleStartElicitation}>
        Learn My Preferences
      </button>
    );
  }

  if (query && query.status === 'eliciting') {
    return (
      <CounterfactualQuestion
        scenarioA={query.scenario_pair.scenario_a}
        scenarioB={query.scenario_pair.scenario_b}
        onChoose={handleAnswer}
        loading={loading}
      />
    );
  }

  if (preferences && preferences.status === 'converged') {
    return (
      <PreferencesDisplay
        weights={preferences.value_function.weights}
        confidence={preferences.value_function.confidence}
        numQueries={preferences.num_queries}
      />
    );
  }

  return null;
}

function CounterfactualQuestion({ scenarioA, scenarioB, onChoose, loading }: QuestionProps) {
  return (
    <div className="preference-question">
      <h3>Which scenario do you prefer?</h3>

      <div className="scenarios">
        <ScenarioCard
          scenario={scenarioA}
          label="A"
          onSelect={() => onChoose('A')}
          disabled={loading}
        />

        <div className="vs">vs</div>

        <ScenarioCard
          scenario={scenarioB}
          label="B"
          onSelect={() => onChoose('B')}
          disabled={loading}
        />
      </div>

      <ProgressBar queriesAnswered={3} totalQueries={10} />
    </div>
  );
}
```

**Exercise:**
1. Trigger preference elicitation
2. Answer 5-10 counterfactual questions
3. See preferences converge
4. Display learned value function

---

### Part 5: Q&A and Next Steps (10 min)

#### Common Integration Patterns

**Pattern 1: Validate → Counterfactual → Robustness → Decide**
```typescript
// Full decision workflow
const validation = await isl.validateCausal({...});
if (validation.status !== 'identifiable') return;

const counterfactual = await isl.generateCounterfactual({...});
const robustness = await isl.analyzeRobustness({
  intervention_proposal: { price: 55 },
  target_outcome: { revenue: [counterfactual.prediction.point_estimate.revenue * 0.95,
                               counterfactual.prediction.point_estimate.revenue * 1.05] }
});

if (robustness.analysis.is_fragile) {
  console.warn('Recommendation is fragile!');
}
```

**Pattern 2: Preferences → Counterfactual → Rank Options**
```typescript
// Learn preferences, then rank scenarios
const prefs = await isl.elicitPreferences({...});
const scenarios = await Promise.all(
  interventions.map(i => isl.generateCounterfactual({ intervention: i, ... }))
);

const ranked = scenarios.sort((a, b) => {
  const scoreA = evaluateWithPreferences(a, prefs.value_function);
  const scoreB = evaluateWithPreferences(b, prefs.value_function);
  return scoreB - scoreA;
});
```

**Pattern 3: Deliberation → Consensus → Validation**
```typescript
// Team deliberates, then validates consensus
const delib = await isl.conductDeliberation({...});
if (delib.consensus_statement.support_score > 0.7) {
  // Validate the agreed intervention
  const validation = await isl.validateCausal({
    ...extractInterventionFromConsensus(delib)
  });
}
```

#### Performance Optimization Tips

1. **Cache responses** for deterministic inputs:
   ```typescript
   const cache = new Map();
   const key = JSON.stringify(request);
   if (cache.has(key)) return cache.get(key);
   ```

2. **Batch requests** when possible:
   ```typescript
   const results = await Promise.all([
     isl.validateCausal({...}),
     isl.generateCounterfactual({...})
   ]);
   ```

3. **Use lower samples** for interactive exploration:
   ```typescript
   // Quick preview: 100 samples
   const preview = await isl.generateCounterfactual({ samples: 100 });

   // Final analysis: 1000 samples
   const final = await isl.generateCounterfactual({ samples: 1000 });
   ```

4. **Monitor rate limits** via headers:
   ```typescript
   const response = await fetch('...', { headers: { 'X-API-Key': key } });
   console.log(response.headers.get('X-RateLimit-Remaining'));
   ```

#### Cost Management Best Practices

1. **Use rule-based fallback** for non-critical paths
2. **Cache LLM responses** aggressively (deterministic prompts)
3. **Monitor daily spend** via Grafana dashboard
4. **Set per-session budgets** conservatively
5. **Review top expensive sessions** weekly

#### Support Channels

- **Documentation:** https://docs.olumi.com/isl
- **Slack:** #isl-integration (for questions)
- **Office Hours:** Wednesdays 2pm GMT
- **Bugs/Issues:** GitHub Issues
- **On-Call:** isl-oncall@olumi.com (critical only)

#### Integration Timeline

**Week 1-2:**
- [ ] Each team integrates ≥2 endpoints
- [ ] TypeScript client working in each codebase
- [ ] Staging environment testing

**Week 3:**
- [ ] End-to-end integration testing
- [ ] Performance validation
- [ ] Cost monitoring setup

**Week 4:**
- [ ] Production deployment
- [ ] Pilot launch with 10 users

---

## Post-Workshop Deliverables

- [ ] Integration plan per team (2-week sprint)
- [ ] Dedicated Slack channel: #isl-integration
- [ ] Weekly sync meetings (Wednesdays 2pm)
- [ ] Integration milestone tracking in Jira

## Success Criteria

By end of workshop:
- [x] Each team successfully calls at least 2 ISL endpoints
- [x] TypeScript client compiling in each team's codebase
- [x] Clear understanding of cost implications
- [x] Integration sprint planned with milestones
- [x] All questions answered

By end of 2-week integration sprint:
- [ ] PLoT validates all causal models via ISL
- [ ] CEE displays assumptions for all causal claims
- [ ] UI has preference learning and deliberation working
- [ ] All teams under rate limits
- [ ] Daily LLM costs <$10
