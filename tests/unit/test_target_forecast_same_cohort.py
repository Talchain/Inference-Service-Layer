"""AIE laws ruling (13 Oct, N_customers / same_cohort): marketing's two targets on ONE cohort.

Conversion above 15% and at least 20 new customers over the first three months after launch. When
both count the same cohort, the customers are N ~ Binomial(E, p) with the cohort's propensity
p ~ Beta (quartiles from an adopted card) and the realised rate is C = N / E on the SAME draw.
Rows go through the real analyze path; the oracle draws with numpy alone and compares with plain
operators (no production fitter, comparator or AND).
"""
import copy
import hashlib
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scipy.stats import beta as beta_dist

from src.api.main import app
from src.models.robustness_v2 import RobustnessRequestV2
from src.services.robustness_analyzer_v2 import RobustnessAnalyzerV2

SEED = 42
N_DRAWS = 2000
FORECASTS = {'hire': (100, (0.12, 0.22)), 'ai': (160, (0.08, 0.14)), 'keep': (60, (0.05, 0.10))}


def payload(with_forecast=('hire', 'ai', 'keep')):
    options = []
    for oid, (exposure, quartiles) in FORECASTS.items():
        option = dict(id=oid, label=oid, interventions={'planning': {'hire': .7, 'ai': .5, 'keep': .3}[oid]})
        if oid in with_forecast:
            option['target_forecast'] = dict(
                law='same_cohort_beta_binomial', exposure=exposure, rate_quartiles=list(quartiles),
                conditions=[dict(constraint_id='conv', quantity='rate', threshold=0.15),
                            dict(constraint_id='cust', quantity='count', threshold=20)])
        options.append(option)
    return dict(
        graph=dict(nodes=[dict(id='planning', kind='factor', label='Planning', observed_state=dict(value=.3)),
                          dict(id='customers', kind='outcome', label='New customers'),
                          dict(id='conversion', kind='goal', label='Conversion rate')],
                   edges=[dict(from_='planning', to='conversion', exists_probability=1., strength=dict(mean=.3, std=.05)),
                          dict(from_='planning', to='customers', exists_probability=1., strength=dict(mean=.3, std=.05))]),
        options=options, goal_node_id='conversion', n_samples=N_DRAWS, seed=SEED, include_voi=False,
        goal_constraints=[dict(constraint_id='conv', node_id='conversion', operator='>', value=.15, value_frame='level'),
                          dict(constraint_id='cust', node_id='customers', operator='>=', value=.1, value_frame='level')],
    )


def run(body):
    return RobustnessAnalyzerV2().analyze(RobustnessRequestV2.model_validate(body))


def oracle(option_id, alpha, beta):
    exposure, _ = FORECASTS[option_id]
    seed = int(hashlib.sha256(f'{SEED}:target_forecast:{option_id}'.encode()).hexdigest()[:8], 16)
    rng = np.random.Generator(np.random.PCG64(seed))
    p = rng.beta(alpha, beta, N_DRAWS)
    n = rng.binomial(exposure, p)
    return n / exposure > 0.15, n >= 20, n


def test_a_both_targets_on_the_same_draws_strict_kept_joint_is_the_share_of_draws_meeting_both():
    response = run(payload())
    for result in response.results:
        tf = result.target_forecast_analysis
        assert tf is not None
        conv, cust, _n = oracle(result.option_id, tf.law['alpha'], tf.law['beta'])
        assert [c.prob_satisfied for c in tf.conditions] == [conv.mean(), cust.mean()]
        assert tf.joint_probability == (conv & cust).mean()
        assert tf.joint_probability_precision == {
            'n_informative': float(N_DRAWS), 'standard_error': math.sqrt(tf.joint_probability * (1 - tf.joint_probability) / N_DRAWS)}
        assert [(c.constraint_id, c.quantity, c.operator, c.threshold) for c in tf.conditions] == [
            ('conv', 'rate', '>', .15), ('cust', 'count', '>=', 20.)]
    hire = next(r for r in response.results if r.option_id == 'hire').target_forecast_analysis
    conv, cust, n = oracle('hire', hire.law['alpha'], hire.law['beta'])
    # Discriminating: with E = 100, a draw of exactly 15 customers is a 15% rate, which is NOT above 15%
    # (strict kept on the same draw), and the two targets move together, so the joint is not the
    # product of the marginals.
    assert (n == 15).sum() > 20 and not conv[n == 15].any()
    assert abs(hire.joint_probability - conv.mean() * cust.mean()) > 0.01


def test_b_the_law_echo_is_the_adopted_quartiles_beta_and_the_point_cohort():
    for result in run(payload()).results:
        law = result.target_forecast_analysis.law
        exposure, (q25, q75) = FORECASTS[result.option_id]
        assert law['family'] == 'beta_binomial' and law['exposure'] == exposure and law['rate_quartiles'] == [q25, q75]
        assert beta_dist.ppf([.25, .75], law['alpha'], law['beta']) == pytest.approx([q25, q75], abs=1e-7)


def test_c_inert_when_absent_and_never_moves_another_option_or_the_model():
    with_one = run(payload(('hire',)))
    without = run(payload(()))
    assert [r.target_forecast_analysis is not None for r in with_one.results] == [True, False, False]
    assert all(r.target_forecast_analysis is None for r in without.results)
    strip = lambda resp: [r.model_dump(exclude={'target_forecast_analysis'}) for r in resp.results]
    assert strip(with_one) == strip(without)


@pytest.mark.parametrize('mutate, message', [
    (lambda o: o['target_forecast']['conditions'].__setitem__(0, dict(constraint_id='nope', quantity='rate', threshold=.15)), 'names no goal_constraints row'),
    (lambda o: o['target_forecast'].__setitem__('rate_quartiles', [.2, .1]), '0 < q25 < q75 < 1'),
    (lambda o: o['target_forecast'].__setitem__('rate_quartiles', [0., .1]), '0 < q25 < q75 < 1'),
    (lambda o: o['target_forecast']['conditions'][0].__setitem__('threshold', 15), 'a rate threshold is a fraction'),
    (lambda o: o['target_forecast']['conditions'].append(dict(constraint_id='conv', quantity='rate', threshold=.2)), 'at most once'),
    (lambda o: o['target_forecast'].__setitem__('law', 'independent'), 'same_cohort_beta_binomial'),
    (lambda o: o['target_forecast'].__setitem__('rho', 0.5), 'Extra inputs are not permitted'),
])
def test_d_validation_negatives(mutate, message):
    body = payload()
    mutate(body['options'][0])
    with pytest.raises(ValidationError, match=message):
        RobustnessRequestV2.model_validate(body)


def test_e_wire_carries_the_analysis_on_the_v2_route():
    client = TestClient(app)
    res = client.post('/api/v1/robustness/analyze/v2', json=payload(('hire',)))
    assert res.status_code == 200, res.text
    options = {o['option_id']: o for o in res.json()['results']}
    tf = options['hire']['target_forecast_analysis']
    assert set(tf) == {'conditions', 'joint_probability', 'joint_probability_precision', 'law'}
    assert 'target_forecast_analysis' not in options['ai'] or options['ai']['target_forecast_analysis'] is None
