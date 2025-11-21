# ISL TypeScript Client

Type-safe TypeScript client for the Inference Service Layer (ISL).

## Installation

```bash
# Copy these files to your project:
# - types.ts
# - client.ts
# - hooks.ts (if using React)
```

## Usage

### Basic Client

```typescript
import { ISLClient } from './client';

const client = new ISLClient(
  'https://isl-staging.onrender.com',
  'your-api-key'
);

// Validate causal model
const validation = await client.validateCausal({
  dag: {
    nodes: ['X', 'Y', 'Z'],
    edges: [['X', 'Y'], ['Z', 'Y']]
  },
  treatment: 'X',
  outcome: 'Y'
});

console.log(validation.status); // 'identifiable'
```

### React Hooks

```typescript
import { useISLClient, useCausalValidation } from './hooks';

function MyComponent() {
  const client = useISLClient(
    'https://isl-staging.onrender.com',
    process.env.ISL_API_KEY!
  );

  const { validate, loading, error, result } = useCausalValidation(client);

  const handleValidate = async () => {
    await validate({
      dag: { nodes: ['X', 'Y'], edges: [['X', 'Y']] },
      treatment: 'X',
      outcome: 'Y'
    });
  };

  if (loading) return <div>Validating...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (result) return <div>Status: {result.status}</div>;

  return <button onClick={handleValidate}>Validate Model</button>;
}
```

## Available Methods

- `validateCausal()` - Y₀-based causal validation
- `generateCounterfactual()` - Counterfactual predictions
- `analyzeRobustness()` - FACET robustness analysis
- `elicitPreferences()` - ActiVA preference learning
- `updatePreferences()` - Update with user choice
- `conductDeliberation()` - Habermas Machine deliberation
- `getDeliberationSession()` - Get session details
- `healthCheck()` - Check service health
- `getMetrics()` - Get Prometheus metrics

## Available Hooks (React)

- `useISLClient()` - Create memoized client instance
- `useCausalValidation()` - Causal validation with state
- `useCounterfactual()` - Counterfactual generation with state
- `useRobustness()` - Robustness analysis with state
- `usePreferences()` - Preference elicitation with state
- `useDeliberation()` - Deliberation with state

## Type Safety

All request and response types are automatically generated from the OpenAPI schema,
providing full type safety and autocomplete in your IDE.

## Error Handling

All methods throw errors on API failures. Use try/catch or React hooks' error state:

```typescript
try {
  const result = await client.validateCausal(request);
} catch (error) {
  console.error('API error:', error.message);
}
```

## Regenerating Types

When the ISL API changes, regenerate the types:

```bash
cd /path/to/inference-service-layer
./scripts/generate_typescript.sh
```
