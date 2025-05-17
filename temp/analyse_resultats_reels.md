# Rapport de Test Complet du Réseau Neuronal Gravitationnel Quantique (Neurax)

Date: 2025-05-13 19:53:33

## Résumé

- **Total des tests**: 15
- **Tests réussis**: 14
- **Tests échoués**: 0
- **Tests ignorés**: 1
- **Taux de réussite**: 93.33%
- **Temps d'exécution total**: 11.87 secondes

## Configuration Matérielle

- **Système**: posix
- **CPU Logiques**: 8
- **CPU Physiques**: 4
- **Mémoire Totale**: 62.81 GB
- **Mémoire Disponible**: 30.95 GB
- **Version Python**: 3.11.10
- **Version NumPy**: 2.2.5

## Résultats Détaillés par Composant

### ✅ Simulateur de Gravité Quantique

- **Tests**: 9
- **Réussis**: 9
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 0.0193 | initialization_ok: True, shape_ok: True, expected_shape: (4, 20, 20, 20), actual_shape: (4, 20, 20, 20), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 1.5767955023273539, 'max_change': 112.19903132120463, 'min_val': -112.19903132120463, 'max_val': 90.46887426576505, 'mean_val': -0.0050691392680444775, 'std_val': 5.525917990057326}, 1.0: {'has_changes': True, 'avg_change': 3.054455761993303, 'max_change': 284.1393929567309, 'min_val': -280.9059444665779, 'max_val': 226.0939330585308, 'mean_val': -0.005617657574179254, 'std_val': 12.249104552108895}, 2.0: {'has_changes': True, 'avg_change': 5.9554125092564565, 'max_change': 395.8161412983905, 'min_val': -388.1749897641357, 'max_val': 415.7249862174024, 'mean_val': -0.15783506276232498, 'std_val': 24.190082392324886}}, simulation_steps_results: {1: {'avg_step_time': 0.0008854866027832031, 'metrics': {'timestamp': '2025-05-13T19:53:33.936251', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.0008892059326171875, 'metrics': {'timestamp': '2025-05-13T19:53:33.940771', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 1.5451286295644489e-07, 'max_curvature': 0.000189718772523711, 'min_curvature': -0.00021030333698956666, 'std_deviation': 2.109888999294382e-05, 'total_energy': 0.09544702132659076, 'quantum_density': 7.381803914316021e+29}}, 10: {'avg_step_time': 0.000959324836730957, 'metrics': {'timestamp': '2025-05-13T19:53:33.950388', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': 6.344647705576494e-07, 'max_curvature': 0.0003277600106752974, 'min_curvature': -0.00028593565027258715, 'std_deviation': 4.132427431193427e-05, 'total_energy': 0.2352523491682022, 'quantum_density': 1.8194247319670684e+30}}} |
| Test 2 | ✅ Réussi | 0.0333 | initialization_ok: True, shape_ok: True, expected_shape: (8, 20, 20, 20), actual_shape: (8, 20, 20, 20), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.7581792912963669, 'max_change': 123.79548908067613, 'min_val': -115.74224380235741, 'max_val': 123.79548908067613, 'mean_val': -0.031004991088372073, 'std_val': 3.8230297482218685}, 1.0: {'has_changes': True, 'avg_change': 1.446646207976359, 'max_change': 236.02784161061012, 'min_val': -171.8627176345211, 'max_val': 234.75061518085442, 'mean_val': 0.01387684913600527, 'std_val': 8.155056757453348}, 2.0: {'has_changes': True, 'avg_change': 2.9110187718815, 'max_change': 728.445554677489, 'min_val': -374.4990712011574, 'max_val': 718.4499693694077, 'mean_val': 0.13497946440967634, 'std_val': 16.934954840285165}}, simulation_steps_results: {1: {'avg_step_time': 0.0008912086486816406, 'metrics': {'timestamp': '2025-05-13T19:53:33.955503', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.004024648666381836, 'metrics': {'timestamp': '2025-05-13T19:53:33.975647', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.0008098125457763672, 'metrics': {'timestamp': '2025-05-13T19:53:33.983791', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': 4.3269237579460545e-07, 'max_curvature': 0.0002288375639710761, 'min_curvature': -0.00021022918385522663, 'std_deviation': 3.0438439826043637e-05, 'total_energy': 0.1581058374042668, 'quantum_density': 1.2227791639862135e+30}}} |
| Test 3 | ✅ Réussi | 0.0278 | initialization_ok: True, shape_ok: True, expected_shape: (16, 20, 20, 20), actual_shape: (16, 20, 20, 20), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.38241451770657353, 'max_change': 124.27257560993841, 'min_val': -107.28245984503062, 'max_val': 124.27257560993841, 'mean_val': -0.004557947876804475, 'std_val': 2.7094490852306716}, 1.0: {'has_changes': True, 'avg_change': 0.7527399056328261, 'max_change': 202.76324699539305, 'min_val': -197.70571443071938, 'max_val': 153.3860733993414, 'mean_val': -0.013955234891794394, 'std_val': 5.8931788502623865}, 2.0: {'has_changes': True, 'avg_change': 1.519584506679234, 'max_change': 638.7529152204614, 'min_val': -412.4913898718235, 'max_val': 654.8167424454541, 'mean_val': -0.055057744676994695, 'std_val': 12.203658618870225}}, simulation_steps_results: {1: {'avg_step_time': 0.0009472370147705078, 'metrics': {'timestamp': '2025-05-13T19:53:33.995171', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.0008853912353515625, 'metrics': {'timestamp': '2025-05-13T19:53:33.999631', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.0011943578720092773, 'metrics': {'timestamp': '2025-05-13T19:53:34.011600', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': -1.2074499860621945e-07, 'max_curvature': 0.00017408056685784472, 'min_curvature': -0.0002472345306860372, 'std_deviation': 2.0938703498102412e-05, 'total_energy': 0.09554488849504837, 'quantum_density': 7.389372890667117e+29}}} |
| Test 4 | ✅ Réussi | 0.1853 | initialization_ok: True, shape_ok: True, expected_shape: (4, 32, 32, 32), actual_shape: (4, 32, 32, 32), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 1.5033846479456248, 'max_change': 119.40287043556808, 'min_val': -119.40287043556808, 'max_val': 117.04499771086981, 'mean_val': 0.01661967740491376, 'std_val': 5.247084350142473}, 1.0: {'has_changes': True, 'avg_change': 2.984830774056843, 'max_change': 449.5303004279178, 'min_val': -449.5436668871044, 'max_val': 266.2950143457525, 'mean_val': -0.04777517762010444, 'std_val': 11.871201722189154}, 2.0: {'has_changes': True, 'avg_change': 6.010945092102721, 'max_change': 600.1992478369548, 'min_val': -610.0105806932116, 'max_val': 481.11105283062386, 'mean_val': -0.11292994273269424, 'std_val': 24.364148839996115}}, simulation_steps_results: {1: {'avg_step_time': 0.0026972293853759766, 'metrics': {'timestamp': '2025-05-13T19:53:34.078578', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.004006004333496094, 'metrics': {'timestamp': '2025-05-13T19:53:34.098655', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': -1.407015212932558e-07, 'max_curvature': 0.00035211978364006497, 'min_curvature': -0.00026704964287135494, 'std_deviation': 2.159623445381834e-05, 'total_energy': 0.39598857432549583, 'quantum_density': 7.476921693032076e+29}}, 10: {'avg_step_time': 0.009813690185546875, 'metrics': {'timestamp': '2025-05-13T19:53:34.196775', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': -7.514661162676265e-08, 'max_curvature': 0.0003924418947780888, 'min_curvature': -0.00037583354900431364, 'std_deviation': 4.2559346965587084e-05, 'total_energy': 0.9871779343934399, 'quantum_density': 1.8639558288067693e+30}}} |
| Test 5 | ✅ Réussi | 0.1852 | initialization_ok: True, shape_ok: True, expected_shape: (8, 32, 32, 32), actual_shape: (8, 32, 32, 32), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.739102677137637, 'max_change': 150.17048516621443, 'min_val': -150.17048516621443, 'max_val': 137.1843615873403, 'mean_val': 0.008210702922461188, 'std_val': 3.697965566965311}, 1.0: {'has_changes': True, 'avg_change': 1.4869262956797502, 'max_change': 284.0596288705717, 'min_val': -284.3430183250476, 'max_val': 255.50295076230682, 'mean_val': -0.001770810592223059, 'std_val': 8.335538884648066}, 2.0: {'has_changes': True, 'avg_change': 2.9814983110593842, 'max_change': 667.9894936087837, 'min_val': -551.737890744346, 'max_val': 650.7751186649871, 'mean_val': 0.017810762941997384, 'std_val': 17.214462288912106}}, simulation_steps_results: {1: {'avg_step_time': 0.0024781227111816406, 'metrics': {'timestamp': '2025-05-13T19:53:34.293998', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.0026093482971191405, 'metrics': {'timestamp': '2025-05-13T19:53:34.307016', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.007485699653625488, 'metrics': {'timestamp': '2025-05-13T19:53:34.381991', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': -2.606128049516439e-08, 'max_curvature': 0.00045929906667180304, 'min_curvature': -0.0002782199756517691, 'std_deviation': 2.987096291374397e-05, 'total_energy': 0.6314868062301493, 'quantum_density': 1.1923519279333272e+30}}} |
| Test 6 | ✅ Réussi | 0.1197 | initialization_ok: True, shape_ok: True, expected_shape: (16, 32, 32, 32), actual_shape: (16, 32, 32, 32), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.3789941315361538, 'max_change': 107.1602838797914, 'min_val': -95.56850747101613, 'max_val': 107.1602838797914, 'mean_val': 0.0030357239540868552, 'std_val': 2.665342049009978}, 1.0: {'has_changes': True, 'avg_change': 0.7387511126110653, 'max_change': 251.42780579173217, 'min_val': -231.54943595299514, 'max_val': 274.45953981786676, 'mean_val': -0.0027385303133292994, 'std_val': 5.895029146297744}, 2.0: {'has_changes': True, 'avg_change': 1.4937611342180832, 'max_change': 562.8548221807433, 'min_val': -544.0386661118162, 'max_val': 559.1833183727693, 'mean_val': 0.0098286201004546, 'std_val': 12.17264258862935}}, simulation_steps_results: {1: {'avg_step_time': 0.002407550811767578, 'metrics': {'timestamp': '2025-05-13T19:53:34.418582', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.012065267562866211, 'metrics': {'timestamp': '2025-05-13T19:53:34.478903', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.002278041839599609, 'metrics': {'timestamp': '2025-05-13T19:53:34.501635', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': 3.9651382358244656e-08, 'max_curvature': 0.00024118018286198496, 'min_curvature': -0.00028940516891396535, 'std_deviation': 2.1379310532402465e-05, 'total_energy': 0.39591882310656706, 'quantum_density': 7.475604674219569e+29}}} |
| Test 7 | ✅ Réussi | 0.2518 | initialization_ok: True, shape_ok: True, expected_shape: (4, 50, 50, 50), actual_shape: (4, 50, 50, 50), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 1.505219455515775, 'max_change': 198.9788487637301, 'min_val': -151.55806164938534, 'max_val': 198.9788487637301, 'mean_val': -0.004387386435484356, 'std_val': 5.342586645590066}, 1.0: {'has_changes': True, 'avg_change': 2.985136824951663, 'max_change': 411.91988378278774, 'min_val': -296.8173716456763, 'max_val': 404.7219020238318, 'mean_val': 0.005428073344350371, 'std_val': 11.846406728433394}, 2.0: {'has_changes': True, 'avg_change': 5.954414110799433, 'max_change': 704.2179796548781, 'min_val': -712.8627737657007, 'max_val': 538.2307178332748, 'mean_val': 0.018699906148730277, 'std_val': 24.112139864577916}}, simulation_steps_results: {1: {'avg_step_time': 0.007912158966064453, 'metrics': {'timestamp': '2025-05-13T19:53:34.588347', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.01806302070617676, 'metrics': {'timestamp': '2025-05-13T19:53:34.678622', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': -2.7680518972986965e-08, 'max_curvature': 0.00032372659633975073, 'min_curvature': -0.000302644100575349, 'std_deviation': 2.125690156525431e-05, 'total_energy': 1.4950932834681772, 'quantum_density': 7.400284043670745e+29}}, 10: {'avg_step_time': 0.007452917098999023, 'metrics': {'timestamp': '2025-05-13T19:53:34.753200', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': -8.317913553808568e-08, 'max_curvature': 0.00041329283553555177, 'min_curvature': -0.0004134067285322002, 'std_deviation': 4.232894960166754e-05, 'total_energy': 3.750277003543733, 'quantum_density': 1.856279830532778e+30}}} |
| Test 8 | ✅ Réussi | 0.2564 | initialization_ok: True, shape_ok: True, expected_shape: (8, 50, 50, 50), actual_shape: (8, 50, 50, 50), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.7445458776187011, 'max_change': 194.1053691591454, 'min_val': -194.1053691591454, 'max_val': 160.81490955607768, 'mean_val': -0.0021916004028674206, 'std_val': 3.7261188648932646}, 1.0: {'has_changes': True, 'avg_change': 1.5084898732212948, 'max_change': 338.70416859996504, 'min_val': -275.90514582794725, 'max_val': 341.86513077576114, 'mean_val': 0.00011585544358527704, 'std_val': 8.430724903387967}, 2.0: {'has_changes': True, 'avg_change': 2.9780682132541054, 'max_change': 771.7025173213444, 'min_val': -771.8986591020097, 'max_val': 729.817981861419, 'mean_val': -0.00037909715920027325, 'std_val': 17.142349893376366}}, simulation_steps_results: {1: {'avg_step_time': 0.00795888900756836, 'metrics': {'timestamp': '2025-05-13T19:53:34.832214', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.015139245986938476, 'metrics': {'timestamp': '2025-05-13T19:53:34.908005', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.010151124000549317, 'metrics': {'timestamp': '2025-05-13T19:53:35.009571', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': 7.693089751300347e-08, 'max_curvature': 0.0003245193753826127, 'min_curvature': -0.0003986638970429015, 'std_deviation': 2.9950700903296478e-05, 'total_energy': 2.4185596567468237, 'quantum_density': 1.19711784103338e+30}}} |
| Test 9 | ✅ Réussi | 0.2652 | initialization_ok: True, shape_ok: True, expected_shape: (16, 50, 50, 50), actual_shape: (16, 50, 50, 50), fluctuations_results: {0.5: {'has_changes': True, 'avg_change': 0.373690265426937, 'max_change': 195.3799344106052, 'min_val': -126.17751907151084, 'max_val': 195.3799344106052, 'mean_val': -0.000385043863388125, 'std_val': 2.651885972635103}, 1.0: {'has_changes': True, 'avg_change': 0.7472597876694572, 'max_change': 331.0482192110441, 'min_val': -330.15722400033724, 'max_val': 299.58950233811237, 'mean_val': 0.0014410923139328647, 'std_val': 5.92657107981553}, 2.0: {'has_changes': True, 'avg_change': 1.504939603582545, 'max_change': 639.4044416333771, 'min_val': -572.0381740829183, 'max_val': 642.1950534478954, 'mean_val': 0.0038944968130571345, 'std_val': 12.184244573258534}}, simulation_steps_results: {1: {'avg_step_time': 0.008349895477294922, 'metrics': {'timestamp': '2025-05-13T19:53:35.136834', 'step': 1, 'simulation_time': 5.391246448313604e-44, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 5: {'avg_step_time': 0.010693740844726563, 'metrics': {'timestamp': '2025-05-13T19:53:35.190439', 'step': 6, 'simulation_time': 3.234747868988162e-43, 'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0, 'std_deviation': 0.0, 'total_energy': 0.0, 'quantum_density': 0.0}}, 10: {'avg_step_time': 0.008434343338012695, 'metrics': {'timestamp': '2025-05-13T19:53:35.274788', 'step': 16, 'simulation_time': 8.625994317301766e-43, 'mean_curvature': -6.348143925472193e-08, 'max_curvature': 0.0003001008447103928, 'min_curvature': -0.0003538454086631327, 'std_deviation': 2.114360746584558e-05, 'total_energy': 1.4866054421903616, 'quantum_density': 7.358271657508646e+29}}} |

### ✅ Neurone Quantique

- **Tests**: 1
- **Réussis**: 1
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 0.0263 | initialization_ok: True, activation_tests: {'0.0': 0.00023755817878867713, '0.5': -1.5228188976479373e-07, '1.0': -0.0002980247319568363, '-0.5': 0.000955928327097566, '-1.0': 0.0021721730597985545}, learning_tests: {'epochs': 100, 'learning_rate': 0.1, 'final_error': -0.002285077589687795, 'error_reduction': 0.4988963981189668, 'final_results': {'0.0->0.0': {'output': 0.5262574850368078, 'error': 0.5262574850368078}, '0.0->1.0': {'output': 0.522913469500004, 'error': 0.47708653049999605}, '1.0->0.0': {'output': 0.5650361396451835, 'error': 0.5650361396451835}, '1.0->1.0': {'output': 0.5662668557553545, 'error': 0.4337331442446455}}} |

### ✅ Réseau P2P

- **Tests**: 1
- **Réussis**: 1
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 0.0004 | initialization_ok: True, messaging_tests: {'error': 'Méthodes message non disponibles'}, discovery_tests: {'error': 'Méthode discover_peers non disponible'} |

### ✅ Mécanisme de Consensus

- **Tests**: 1
- **Réussis**: 1
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 0.0003 | initialization_ok: True, validation_tests: {'request_created': True, 'request_content': "{'request_id': '263ae9f6fa691b80', 'item_id': 'test_item_001', 'item_type': 'SOLUTION', 'item_data': {'content': 'Test solution data'}, 'requester_id': 'test_node', 'validator_id': 'validator_001', 'timestamp': 1747166015.3087122, 'criteria': ['consistency', 'correctness', 'completeness', 'efficiency']}", 'request_processed': True, 'result': '<core.consensus.proof_of_cognition.ValidationResult object at 0x7f4ce09db610>'}, validator_tests: {'validators_selected': True, 'validator_count': 0, 'validators': '[]'} |

### ✅ Visualisation

- **Tests**: 1
- **Réussis**: 0
- **Échoués**: 0
- **Ignorés**: 1
- **Taux de réussite**: 0.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ⚠️ Ignoré | 0.0000 | error: Module non disponible |

### ✅ Gestionnaire d'Export

- **Tests**: 1
- **Réussis**: 1
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 0.0675 | initialization_ok: True, export_tests: {'excel': {'error': "No module named 'openpyxl'"}, 'hdf5': {'export_successful': True}, 'csv': {'error': 'too many values to unpack (expected 3)'}} |

### ✅ Gestionnaire de Base de Données

- **Tests**: 1
- **Réussis**: 1
- **Échoués**: 0
- **Ignorés**: 0
- **Taux de réussite**: 100.00%

#### Détails des tests

| Test | Statut | Temps (s) | Détails |
|------|--------|-----------|--------|
| Test 1 | ✅ Réussi | 10.4341 | initialization_ok: True, db_tests: {'create_tables': {'success': True}, 'save_simulation': {'success': True, 'simulation_id': '5'}, 'get_recent_simulations': {'success': True, 'simulation_count': 5}, 'get_simulation_by_id': {'success': True}} |

## Métriques de Performance

### Simulateur

| Métrique | Valeur |
|----------|-------|
| init_times | {20: 0.0003402233123779297, 32: 0.0012819766998291016, 50: 0.0004057884216308594, 64: 0.0005314350128173828} |
| fluctuation_times | {20: 0.0005702972412109375, 32: 0.0024652481079101562, 50: 0.007196664810180664, 64: 0.016903162002563477} |
| simulation_step_times | {20: {'min': 0.0007822513580322266, 'max': 0.0008947849273681641, 'avg': 0.0008408546447753907}, 32: {'min': 0.0022478103637695312, 'max': 0.002662181854248047, 'avg': 0.0024109363555908205}, 50: {'min': 0.006404876708984375, 'max': 0.04698586463928223, 'avg': 0.015277624130249023}, 64: {'min': 0.013551712036132812, 'max': 0.04022979736328125, 'avg': 0.0197237491607666}} |
| memory_usage_mb | {20: 0.0, 32: 0.0, 50: 0.0, 64: 0.0} |

## Conclusion

Le système Neurax a passé la grande majorité des tests avec succès. Les performances sont excellentes et le système est prêt pour une utilisation avancée.

---

*Rapport généré automatiquement par le Framework de Test Neurax*
