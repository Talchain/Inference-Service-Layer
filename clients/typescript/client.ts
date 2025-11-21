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
