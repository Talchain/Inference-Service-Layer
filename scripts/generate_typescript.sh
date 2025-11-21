#!/bin/bash
# Generate TypeScript types and client from OpenAPI schema

set -e

echo "🔧 Generating TypeScript client for ISL..."
echo

# Step 1: Generate OpenAPI schema
echo "📋 Step 1: Generating OpenAPI schema..."
cd "$(dirname "$0")/.."
poetry run python scripts/generate_openapi.py
echo

# Step 2: Check if npx is available
if ! command -v npx &> /dev/null; then
    echo "❌ Error: npx not found. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Step 3: Generate TypeScript types
echo "📝 Step 2: Generating TypeScript types..."
npx openapi-typescript openapi.json --output clients/typescript/types.ts
echo "✓ Types generated: clients/typescript/types.ts"
echo

# Step 4: Generate TypeScript client
echo "🔨 Step 3: Generating TypeScript client..."
cat > clients/typescript/client.ts << 'EOF'
/**
 * ISL TypeScript Client
 *
 * Auto-generated from OpenAPI schema.
 * Provides type-safe access to all ISL endpoints.
 */

import type { paths } from './types';

type FetchOptions = RequestInit & {
  params?: Record<string, string>;
};

export class ISLClient {
  constructor(
    private baseUrl: string,
    private apiKey: string
  ) {
    // Remove trailing slash from baseUrl
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  /**
   * Make authenticated request to ISL API
   */
  private async request<T>(
    path: string,
    options: FetchOptions = {}
  ): Promise<T> {
    const { params, ...fetchOptions } = options;

    // Build URL with query params
    let url = `${this.baseUrl}${path}`;
    if (params) {
      const searchParams = new URLSearchParams(params);
      url += `?${searchParams.toString()}`;
    }

    // Add auth header
    const headers = new Headers(fetchOptions.headers);
    headers.set('X-API-Key', this.apiKey);
    if (fetchOptions.method === 'POST' || fetchOptions.method === 'PUT') {
      headers.set('Content-Type', 'application/json');
    }

    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`ISL API error (${response.status}): ${errorText}`);
    }

    return response.json();
  }

  /**
   * Validate causal model
   */
  async validateCausal(
    request: paths['/api/v1/causal/validate']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/causal/validate']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/causal/validate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Generate counterfactual prediction
   */
  async generateCounterfactual(
    request: paths['/api/v1/causal/counterfactual']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/causal/counterfactual']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/causal/counterfactual', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Analyze robustness of intervention
   */
  async analyzeRobustness(
    request: paths['/api/v1/robustness/analyze']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/robustness/analyze']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/robustness/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Elicit user preferences (ActiVA)
   */
  async elicitPreferences(
    request: paths['/api/v1/preferences/elicit']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/preferences/elicit']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/preferences/elicit', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Update user preferences with choice
   */
  async updatePreferences(
    request: paths['/api/v1/preferences/update']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/preferences/update']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/preferences/update', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Conduct deliberation round (Habermas Machine)
   */
  async conductDeliberation(
    request: paths['/api/v1/deliberation/deliberate']['post']['requestBody']['content']['application/json']
  ): Promise<paths['/api/v1/deliberation/deliberate']['post']['responses']['200']['content']['application/json']> {
    return this.request('/api/v1/deliberation/deliberate', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  /**
   * Get deliberation session details
   */
  async getDeliberationSession(
    sessionId: string
  ): Promise<any> {
    return this.request(`/api/v1/deliberation/session/${sessionId}`, {
      method: 'GET',
    });
  }

  /**
   * Check health status
   */
  async healthCheck(): Promise<any> {
    return this.request('/health', {
      method: 'GET',
    });
  }

  /**
   * Get Prometheus metrics
   */
  async getMetrics(): Promise<string> {
    const response = await fetch(`${this.baseUrl}/metrics`, {
      headers: {
        'X-API-Key': this.apiKey,
      },
    });
    return response.text();
  }
}

export default ISLClient;
EOF
echo "✓ Client generated: clients/typescript/client.ts"
echo

# Step 5: Generate React hooks
echo "⚛️  Step 4: Generating React hooks..."
cat > clients/typescript/hooks.ts << 'EOF'
/**
 * React hooks for ISL integration
 *
 * Provides easy-to-use hooks for React applications.
 */

import { useState, useCallback, useEffect } from 'react';
import { ISLClient } from './client';
import type { paths } from './types';

type CausalValidationRequest = paths['/api/v1/causal/validate']['post']['requestBody']['content']['application/json'];
type CausalValidationResponse = paths['/api/v1/causal/validate']['post']['responses']['200']['content']['application/json'];

type CounterfactualRequest = paths['/api/v1/causal/counterfactual']['post']['requestBody']['content']['application/json'];
type CounterfactualResponse = paths['/api/v1/causal/counterfactual']['post']['responses']['200']['content']['application/json'];

type RobustnessRequest = paths['/api/v1/robustness/analyze']['post']['requestBody']['content']['application/json'];
type RobustnessResponse = paths['/api/v1/robustness/analyze']['post']['responses']['200']['content']['application/json'];

type PreferencesRequest = paths['/api/v1/preferences/elicit']['post']['requestBody']['content']['application/json'];
type PreferencesResponse = paths['/api/v1/preferences/elicit']['post']['responses']['200']['content']['application/json'];

type DeliberationRequest = paths['/api/v1/deliberation/deliberate']['post']['requestBody']['content']['application/json'];
type DeliberationResponse = paths['/api/v1/deliberation/deliberate']['post']['responses']['200']['content']['application/json'];

/**
 * Create ISL client instance (memoized)
 */
export function useISLClient(baseUrl: string, apiKey: string) {
  const [client] = useState(() => new ISLClient(baseUrl, apiKey));
  return client;
}

/**
 * Hook for causal validation
 */
export function useCausalValidation(client: ISLClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<CausalValidationResponse | null>(null);

  const validate = useCallback(async (request: CausalValidationRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.validateCausal(request);
      setResult(response);
      return response;
    } catch (e) {
      const error = e as Error;
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { validate, loading, error, result };
}

/**
 * Hook for counterfactual generation
 */
export function useCounterfactual(client: ISLClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<CounterfactualResponse | null>(null);

  const generate = useCallback(async (request: CounterfactualRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.generateCounterfactual(request);
      setResult(response);
      return response;
    } catch (e) {
      const error = e as Error;
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { generate, loading, error, result };
}

/**
 * Hook for robustness analysis
 */
export function useRobustness(client: ISLClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<RobustnessResponse | null>(null);

  const analyze = useCallback(async (request: RobustnessRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.analyzeRobustness(request);
      setResult(response);
      return response;
    } catch (e) {
      const error = e as Error;
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { analyze, loading, error, result };
}

/**
 * Hook for preference elicitation (ActiVA)
 */
export function usePreferences(client: ISLClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<PreferencesResponse | null>(null);

  const elicit = useCallback(async (request: PreferencesRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.elicitPreferences(request);
      setResult(response);
      return response;
    } catch (e) {
      const error = e as Error;
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { elicit, loading, error, result };
}

/**
 * Hook for deliberation (Habermas Machine)
 */
export function useDeliberation(client: ISLClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [result, setResult] = useState<DeliberationResponse | null>(null);

  const deliberate = useCallback(async (request: DeliberationRequest) => {
    setLoading(true);
    setError(null);
    try {
      const response = await client.conductDeliberation(request);
      setResult(response);
      return response;
    } catch (e) {
      const error = e as Error;
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { deliberate, loading, error, result };
}
EOF
echo "✓ Hooks generated: clients/typescript/hooks.ts"
echo

# Step 6: Generate README
echo "📚 Step 5: Generating TypeScript client README..."
cat > clients/typescript/README.md << 'EOF'
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
EOF
echo "✓ README generated: clients/typescript/README.md"
echo

echo "✅ TypeScript client generation complete!"
echo
echo "📦 Generated files:"
echo "   - clients/typescript/types.ts    (OpenAPI types)"
echo "   - clients/typescript/client.ts   (ISL client class)"
echo "   - clients/typescript/hooks.ts    (React hooks)"
echo "   - clients/typescript/README.md   (Documentation)"
echo
echo "🚀 Next steps:"
echo "   1. Copy clients/typescript/* to your TypeScript project"
echo "   2. See clients/typescript/README.md for usage examples"
echo "   3. Enjoy type-safe ISL integration!"
