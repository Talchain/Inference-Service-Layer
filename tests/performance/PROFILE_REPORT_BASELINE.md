# ISL Performance Profile Baseline

_Generated: 2025-11-22 00:01:46_


## Causal Validation


### Simple

- Iterations: 50
- P50: 2.1ms
- P95: 2.3ms
- P99: 2.6ms
- Mean: 2.1ms
- Min: 2.0ms
- Max: 2.6ms

#### Top Hotspots
```
         213000 function calls (209400 primitive calls) in 0.103 seconds

   Ordered by: cumulative time
   List reduced from 225 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       50    0.000    0.000    0.106    0.002 /home/user/Inference-Service-Layer/tests/performance/profile_endpoints.py:100(run_simple)
       50    0.001    0.000    0.105    0.002 /home/user/Inference-Service-Layer/tests/performance/../../src/services/causal_validator.py:55(validate)
       50    0.001    0.000    0.098    0.002 /home/user/Inference-Service-Layer/tests/performance/../../src/services/causal_validator.py:200(_try_comprehensive_y0_identification)
       50    0.000    0.000    0.088    0.002 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/api.py:16(identify_outcomes)
   200/50    0.002    0.000    0.080    0.002 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:16(identify)
   150/50    0.000    0.000    0.050    0.001 /usr/local/lib/python3.11/dist-packages/y0/dsl.py:1032(safe)
  350/150    0.001    0.000    0.049    0.000 /usr/local/lib/python3.11/dist-packages/y0/dsl.py:1059(<genexpr>)
      600    0.001    0.000    0.028    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:397(from_edges)
      200    0.000    0.000    0.020    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:201(__init__)
      200    0.001    0.000    0.018    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:329(str_nodes_to_variable_nodes)
      150    0.000    0.000    0.017    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:520(get_no_effect_on_outcomes)
      150    0.000    0.000    0.016    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:511(get_intervened_ancestors)
       50    0.000    0.000    0.015    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:161(line_4)
      150    0.000    0.000    0.014    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:217(from_parts)
      800    0.002    0.000    0.014    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:126(add_directed_edge)
2050/1150    0.002    0.000    0.014    0.000 /usr/local/lib/python3.11/dist-packages/networkx/utils/backends.py:525(_call_if_no_backends_installed)
      400    0.000    0.000    0.013    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:555(ancestors_inclusive)
    32650    0.009    0.000    0.012    0.000 <string>:2(__hash__)
     1300    0.002    0.000    0.012    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:120(add_node)
```


### Complex

- Iterations: 30
- P50: 9.0ms
- P95: 11.5ms
- P99: 12.0ms
- Mean: 9.3ms
- Min: 8.7ms
- Max: 12.0ms

#### Top Hotspots
```
         653190 function calls (647340 primitive calls) in 0.270 seconds

   Ordered by: cumulative time
   List reduced from 227 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       30    0.000    0.000    0.280    0.009 /home/user/Inference-Service-Layer/tests/performance/profile_endpoints.py:114(run_complex)
       30    0.000    0.000    0.279    0.009 /home/user/Inference-Service-Layer/tests/performance/../../src/services/causal_validator.py:55(validate)
       30    0.000    0.000    0.272    0.009 /home/user/Inference-Service-Layer/tests/performance/../../src/services/causal_validator.py:200(_try_comprehensive_y0_identification)
       30    0.000    0.000    0.258    0.009 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/api.py:16(identify_outcomes)
   420/30    0.004    0.000    0.246    0.008 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:16(identify)
   180/30    0.001    0.000    0.145    0.005 /usr/local/lib/python3.11/dist-packages/y0/dsl.py:1032(safe)
  540/240    0.001    0.000    0.143    0.001 /usr/local/lib/python3.11/dist-packages/y0/dsl.py:1059(<genexpr>)
      990    0.004    0.000    0.122    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:397(from_edges)
      420    0.000    0.000    0.094    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:201(__init__)
      420    0.001    0.000    0.092    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:329(str_nodes_to_variable_nodes)
      390    0.001    0.000    0.089    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/utils.py:217(from_parts)
       30    0.000    0.000    0.078    0.003 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:161(line_4)
     4500    0.009    0.000    0.075    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:126(add_directed_edge)
       30    0.000    0.000    0.067    0.002 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:185(<listcomp>)
      180    0.001    0.000    0.060    0.000 /usr/local/lib/python3.11/dist-packages/y0/algorithm/identify/id_std.py:99(line_2)
   131550    0.036    0.000    0.048    0.000 <string>:2(__hash__)
    13410    0.019    0.000    0.043    0.000 /usr/local/lib/python3.11/dist-packages/networkx/classes/graph.py:524(add_node)
     4410    0.005    0.000    0.043    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:120(add_node)
      180    0.000    0.000    0.035    0.000 /usr/local/lib/python3.11/dist-packages/y0/graph.py:520(get_no_effect_on_outcomes)
```


## Counterfactual


### Small

- Iterations: 30
- P50: 1.5ms
- P95: 1.6ms
- P99: 1.8ms
- Mean: 1.5ms
- Min: 1.4ms
- Max: 1.8ms

#### Top Hotspots
```
         21301 function calls in 0.044 seconds

   Ordered by: cumulative time
   List reduced from 145 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       30    0.000    0.000    0.045    0.001 /home/user/Inference-Service-Layer/tests/performance/profile_endpoints.py:143(run_small)
       30    0.000    0.000    0.044    0.001 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:49(analyze)
       30    0.000    0.000    0.022    0.001 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:255(_compute_prediction)
      120    0.001    0.000    0.019    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3992(percentile)
      150    0.000    0.000    0.019    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3763(_ureduce)
      120    0.000    0.000    0.016    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4547(_quantile_unchecked)
      120    0.000    0.000    0.016    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4697(_quantile_ureduce_func)
      120    0.002    0.000    0.015    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4765(_quantile)
       30    0.000    0.000    0.010    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:158(_run_monte_carlo)
       30    0.000    0.000    0.007    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:195(_sample_distribution)
       30    0.007    0.000    0.007    0.000 {method 'normal' of 'numpy.random.mtrand.RandomState' objects}
      150    0.007    0.000    0.007    0.000 {method 'partition' of 'numpy.ndarray' objects}
       30    0.001    0.000    0.007    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:295(_analyze_uncertainty)
       90    0.002    0.000    0.004    0.000 /usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py:135(_var)
       60    0.000    0.000    0.003    0.000 /usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py:3654(var)
       30    0.000    0.000    0.003    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3845(median)
       30    0.000    0.000    0.003    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3931(_median)
      120    0.001    0.000    0.002    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4730(_get_indexes)
      120    0.001    0.000    0.002    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4565(_quantile_is_valid)
```


### Large

- Iterations: 20
- P50: 1.7ms
- P95: 2.3ms
- P99: 2.3ms
- Mean: 1.8ms
- Min: 1.6ms
- Max: 2.3ms

#### Top Hotspots
```
         15121 function calls in 0.034 seconds

   Ordered by: cumulative time
   List reduced from 145 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       20    0.000    0.000    0.035    0.002 /home/user/Inference-Service-Layer/tests/performance/profile_endpoints.py:165(run_large)
       20    0.000    0.000    0.034    0.002 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:49(analyze)
       20    0.000    0.000    0.016    0.001 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:255(_compute_prediction)
       80    0.000    0.000    0.014    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3992(percentile)
      100    0.000    0.000    0.014    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3763(_ureduce)
       80    0.000    0.000    0.012    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4547(_quantile_unchecked)
       80    0.000    0.000    0.011    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4697(_quantile_ureduce_func)
       80    0.001    0.000    0.011    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:4765(_quantile)
       20    0.000    0.000    0.009    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:158(_run_monte_carlo)
       20    0.000    0.000    0.005    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:195(_sample_distribution)
       20    0.005    0.000    0.005    0.000 {method 'normal' of 'numpy.random.mtrand.RandomState' objects}
       20    0.001    0.000    0.005    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:295(_analyze_uncertainty)
      100    0.005    0.000    0.005    0.000 {method 'partition' of 'numpy.ndarray' objects}
       80    0.000    0.000    0.003    0.000 /home/user/Inference-Service-Layer/tests/performance/../../src/services/counterfactual_engine.py:220(_evaluate_equation)
       60    0.002    0.000    0.003    0.000 /usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py:135(_var)
       80    0.001    0.000    0.003    0.000 {built-in method builtins.eval}
       40    0.000    0.000    0.002    0.000 /usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py:3654(var)
       20    0.000    0.000    0.002    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3845(median)
       20    0.000    0.000    0.002    0.000 /usr/local/lib/python3.11/dist-packages/numpy/lib/function_base.py:3931(_median)
```


## Bottleneck Analysis

✅ No major bottlenecks identified - all endpoints within targets
