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
