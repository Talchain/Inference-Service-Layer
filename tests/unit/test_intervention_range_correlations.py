"""ISL2: synthetic positive continuous forecasts, not marketing percentage/count laws."""
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2
from src.validation.request_validator import detect_identical_options

SEED = 42
N = 1000
IDS = ['x', 'y', 'z']
Z75 = 0.6744897501960817


def payload(declared=True):
    options = []
    for oid, low, high, rho in [('a', .2, .8, .5), ('b', .3, 1.2, 0.)]:
        option = dict(id=oid, label=oid,
                      interventions={n: math.sqrt(low * high) for n in IDS},
                      intervention_ranges={n: dict(low=low, high=high, meaning='likely_range') for n in IDS})
        if declared:
            option['intervention_range_correlations'] = [
                dict(factor_a=a, factor_b=b, rho=rho) for a, b in itertools.combinations(IDS, 2)
            ]
        options.append(option)
    return dict(
        graph=dict(nodes=[dict(id=n, kind='factor', label=n, observed_state=dict(value=.4)) for n in IDS]
                   + [dict(id='g', kind='goal', label='Goal')],
                   edges=[dict(from_=n, to='g', exists_probability=1., strength=dict(mean=.1, std=.01)) for n in IDS]),
        options=options, goal_node_id='g', n_samples=N, seed=SEED, include_voi=False,
        goal_constraints=[dict(constraint_id=n, node_id=n, operator=op, value=t, value_frame='level')
                          for n, op, t in zip(IDS, ['>', '>=', '<='], [.45, .5, .55])],
    )


def run(body):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(body))


def marginals(response):
    return {r.option_id: [c.model_dump(mode='json') for c in r.constraint_analysis.constraints]
            for r in response.results}


def oracle(option):
    # Independent oracle: no production fitter, copula sampler, series resolver,
    # comparator or AND calculator is called. Three scalar normals per draw.
    key = f"{SEED}:intervention_range_joint:{option['id']}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    rho = option['intervention_range_correlations'][0]['rho']
    matrix = np.full((3, 3), rho)
    np.fill_diagonal(matrix, 1.)
    chol = np.linalg.cholesky(matrix)
    normals = np.array([chol @ np.array([rng.normal(0., 1.) for _ in IDS]) for _ in range(N)])
    levels = np.empty_like(normals)
    for i, n in enumerate(IDS):
        bounds = option['intervention_ranges'][n]
        sigma = (math.log(bounds['high']) - math.log(bounds['low'])) / (2 * Z75)
        mu = math.log(bounds['low']) + Z75 * sigma
        # Preserve the existing fitter's arithmetic order and scalar transform.
        levels[:, i] = [float(np.exp(mu + sigma * z)) for z in normals[:, i]]
    met = np.column_stack([levels[:, 0] > .45, levels[:, 1] >= .5, levels[:, 2] <= .55])
    return levels, met


def assert_withheld(response, ids=('a', 'b')):
    warnings = [w for w in response.inference_warnings if w.code == 'CONSTRAINT_JOINT_WITHHELD']
    assert {w.detail['option_id'] for w in warnings} == set(ids)
    assert all(w.detail['reason'] == 'intervention_range_joint_deferred' for w in warnings)
    for result in response.results:
        if result.option_id in ids:
            assert result.constraint_analysis.joint_probability is None
            assert result.constraint_analysis.joint_probability_precision is None
            assert result.constraint_analysis.conditional_probabilities is None


def test_a_declared_same_population_joint_and_precision():
    body = payload()
    response = run(body)
    series = RobustnessAnalyzerV2._intervention_range_series(RobustnessRequestV2.model_validate(body), SEED, set(IDS))
    for option, result in zip(body['options'], response.results):
        levels, met = oracle(option)
        # Exact draw alignment, beyond the aggregate probability assertion.
        np.testing.assert_array_equal(np.array([series[option['id']][n] for n in IDS]).T, levels)
        p = int(np.all(met, axis=1).sum()) / N
        analysis = result.constraint_analysis
        assert analysis.joint_probability == p
        assert analysis.joint_probability_precision.model_dump() == {
            'n_informative': N, 'standard_error': math.sqrt(p * (1 - p) / N),
        }
        for i, row in enumerate(analysis.constraints):
            assert row.prob_satisfied == int(met[:, i].sum()) / N
        assert [r.model_dump() for r in result.sampled_intervention_ranges] == [
            dict(node_id=n, meaning='likely_range', family='lognormal', coverage=.5,
                 low=option['intervention_ranges'][n]['low'], high=option['intervention_ranges'][n]['high'])
            for n in IDS
        ]
    assert not [w for w in response.inference_warnings if w.code == 'CONSTRAINT_JOINT_WITHHELD']


def test_b_undeclared_withholds_and_preserves_baseline_marginals():
    response = run(payload(False))
    assert_withheld(response)
    golden = Path(__file__).with_name('fixtures') / 'isl2_undeclared_marginals.json'
    assert marginals(response) == json.loads(golden.read_text())
    assert (json.dumps(marginals(response), indent=2, sort_keys=True) + '\n').encode() == golden.read_bytes()


def test_c_partial_declaration_withholds_and_preserves_baseline_marginals():
    body = payload()
    for option in body['options']:
        option['intervention_range_correlations'] = option['intervention_range_correlations'][:1]
    response = run(body)
    assert_withheld(response)
    assert marginals(response) == marginals(run(payload(False)))


def test_one_options_declaration_cannot_license_other_option():
    body = payload()
    body['options'][1].pop('intervention_range_correlations')
    response = run(body)
    assert_withheld(response, ('b',))
    assert response.results[0].constraint_analysis.joint_probability is not None
    assert marginals(response)['b'] == marginals(run(payload(False)))['b']


def test_mixed_range_point_keeps_legacy_series_and_withholding():
    body = payload()
    body['goal_constraints'].append(dict(constraint_id='point', node_id='p', operator='>=', value=.2, value_frame='level'))
    body['graph']['nodes'].append(dict(id='p', kind='factor', label='Point', observed_state=dict(value=.3)))
    for option in body['options']:
        option['interventions']['p'] = .3
    response = run(body)
    assert_withheld(response)
    for option in body['options']:
        option.pop('intervention_range_correlations')
    assert marginals(response) == marginals(run(body))


@pytest.mark.parametrize('pairs,match', [
    ([dict(factor_a='x', factor_b='unknown', rho=0.)], 'both factors'),
    ([dict(factor_a='x', factor_b='y', rho=1.01)], 'less than or equal'),
    ([dict(factor_a='x', factor_b='y', rho=-1.01)], 'greater than or equal'),
    ([dict(factor_a='x', factor_b='y', rho=float('nan'))], 'finite'),
    ([dict(factor_a='x', factor_b='x', rho=0.)], 'self-pairs'),
    ([dict(factor_a='x', factor_b='y', rho=0.), dict(factor_a='y', factor_b='x', rho=0.)], 'duplicate'),
    ([dict(factor_a='x', factor_b='y', rho=1.), dict(factor_a='x', factor_b='z', rho=1.),
      dict(factor_a='y', factor_b='z', rho=-1.)], 'positive semidefinite'),
    ([dict(factor_a='x', factor_b='y', rho=-.51), dict(factor_a='x', factor_b='z', rho=-.51),
      dict(factor_a='y', factor_b='z', rho=-.51)], 'positive semidefinite'),
])
def test_d_invalid_declarations_rejected(pairs, match):
    body = payload()
    body['options'][0]['intervention_range_correlations'] = pairs
    with pytest.raises(ValidationError, match=match):
        RobustnessRequestV2.model_validate(body)


def test_declaration_cannot_borrow_range_from_another_option():
    body = payload()
    del body['options'][0]['intervention_ranges']['z']
    with pytest.raises(ValidationError, match='both factors'):
        RobustnessRequestV2.model_validate(body)


def test_singular_psd_is_admitted_and_runs():
    body = payload()
    for p in body['options'][0]['intervention_range_correlations']:
        p['rho'] = 1.
    assert run(body).results[0].constraint_analysis.joint_probability is not None


def test_declaration_order_and_other_option_do_not_move_draws():
    body = payload()
    before = run(body)
    for option in body['options']:
        option['intervention_range_correlations'].reverse()
        for p in option['intervention_range_correlations']:
            p['factor_a'], p['factor_b'] = p['factor_b'], p['factor_a']
    body['options'].reverse()
    after = run(body)
    assert marginals(before) == marginals(after)
    assert {r.option_id: r.constraint_analysis.model_dump() for r in before.results} == {
        r.option_id: r.constraint_analysis.model_dump() for r in after.results}


def test_api_transport_and_distinct_dependence_options():
    body = payload()
    # Identical point/range forecasts are still distinct when dependence differs.
    body['options'][1]['interventions'] = copy.deepcopy(body['options'][0]['interventions'])
    body['options'][1]['intervention_ranges'] = copy.deepcopy(body['options'][0]['intervention_ranges'])
    assert detect_identical_options(body['options']) is None
    request = RobustnessRequestV2.model_validate(body)
    response = TestClient(app).post('/api/v1/robustness/analyze/v2',
        json=request.model_dump(mode='json', by_alias=True, exclude_none=True),
        headers={'X-ISL-Response-Version': '2'})
    assert response.status_code == 200, response.text
    expected = run(body)
    for wire, result in zip(response.json()['options'], expected.results):
        assert wire['constraint_analysis']['joint_probability'] == result.constraint_analysis.joint_probability
        assert wire['constraint_analysis']['joint_probability_precision'] == result.constraint_analysis.joint_probability_precision.model_dump()
        assert wire['sampled_intervention_ranges'] == [s.model_dump() for s in result.sampled_intervention_ranges]
    same = copy.deepcopy(body['options'])
    same[1]['intervention_range_correlations'] = copy.deepcopy(same[0]['intervention_range_correlations'])
    same[1]['intervention_range_correlations'].reverse()
    for pair in same[1]['intervention_range_correlations']:
        pair['factor_a'], pair['factor_b'] = pair['factor_b'], pair['factor_a']
    assert detect_identical_options(same) == ('a', 'b')


@pytest.mark.parametrize('declaration', [None, []])
def test_empty_or_null_carrier_is_not_declared_independence(declaration):
    body = payload(False)
    for option in body['options']:
        option['intervention_range_correlations'] = declaration
    response = run(body)
    assert_withheld(response)
    assert marginals(response) == marginals(run(payload(False)))


def test_unrelated_pairs_on_same_option_do_not_license_scored_targets():
    body = payload()
    body['graph']['nodes'] += [dict(id=n, kind='factor', label=n) for n in ['r', 's']]
    for option in body['options']:
        for n in ['r', 's']:
            option['interventions'][n] = .4
            option['intervention_ranges'][n] = dict(low=.2, high=.8, meaning='likely_range')
        option['intervention_range_correlations'] = [dict(factor_a='r', factor_b='s', rho=0.)]
    response = run(body)
    assert_withheld(response)
    assert marginals(response) == marginals(run(payload(False)))


@pytest.mark.parametrize('route', ['/api/v1/robustness/analyze/v2', '/api/v1/robustness/analyze/unified'])
def test_legacy_api_and_worker_preserve_carrier_and_joint(route):
    from src.services.robustness_worker import run_robustness_v2, decode_analysis_response

    body = payload()
    request = RobustnessRequestV2.model_validate(body)
    expected = run(body)
    # The actual worker entry point round-trips request/response JSON.
    worker_result = decode_analysis_response(run_robustness_v2(request.model_dump_json()))
    assert [r.constraint_analysis.model_dump() for r in worker_result.results] == [
        r.constraint_analysis.model_dump() for r in expected.results]
    response = TestClient(app).post(route,
        json=request.model_dump(mode='json', by_alias=True, exclude_none=True),
        headers={'X-ISL-Response-Version': '1'})
    assert response.status_code == 200, response.text
    assert [r['constraint_analysis'] for r in response.json()['results']] == [
        r.constraint_analysis.model_dump(mode='json') for r in expected.results]


def test_declared_levels_use_each_ranges_affine_frame():
    body = payload()
    reference = run(body)
    for i, n in enumerate(IDS):
        zero, one = -.1 * i, 2. + i
        for option in body['options']:
            option['intervention_ranges'][n]['normalisation'] = dict(raw_at_zero=zero, raw_at_one=one)
            option['interventions'][n] = (option['interventions'][n] - zero) / (one - zero)
        body['goal_constraints'][i]['value'] = (body['goal_constraints'][i]['value'] - zero) / (one - zero)
    actual = run(body)
    for before, after in zip(reference.results, actual.results):
        assert after.constraint_analysis.joint_probability == before.constraint_analysis.joint_probability
        assert after.constraint_analysis.joint_probability_precision == before.constraint_analysis.joint_probability_precision
        assert [c.prob_satisfied for c in after.constraint_analysis.constraints] == [
            c.prob_satisfied for c in before.constraint_analysis.constraints]
        assert after.sampled_intervention_ranges == before.sampled_intervention_ranges


def test_refused_condition_cannot_be_dropped_to_license_subset_joint():
    body = payload()
    body['goal_constraints'][2]['value_frame'] = 'change_abs'
    response = run(body)
    assert_withheld(response)
    for option in body['options']:
        option.pop('intervention_range_correlations')
    assert marginals(response) == marginals(run(body))
    for result in response.results:
        assert [c.node_id for c in result.constraint_analysis.constraints] == ['x', 'y']


def test_change_quantity_does_not_acquire_absolute_forecast_joint():
    body = payload()
    body['graph']['nodes'][0]['quantity_frame'] = 'change'
    response = run(body)
    assert_withheld(response)
    for option in body['options']:
        option.pop('intervention_range_correlations')
    assert marginals(response) == marginals(run(body))
