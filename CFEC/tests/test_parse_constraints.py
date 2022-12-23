from cfec.constraints import json_to_class_parse, ValueRange, ValueMonotonicity, ValueNominal, ValueMaxDiff


def test_parse():
    dict_constraints = {'range': {7619408953831620358: {'columns': [0, 2, 1],
                                                        'constraints': {'min_value': 0.0, 'max_value': 1.0}},
                                  -5232989938126879403: {'columns': [3],
                                                         'constraints': {'min_value': 10.0, 'max_value': 20.0}}},
                        'nominal': {1435837555717346218: {'columns': [3],
                                                          'constraints': {'value': 'red', 'value.1': 'green',
                                                                          'value.2': 'blue'}},
                                    -68893908727300965: {'columns': [7],
                                                         'constraints': {'value': 'red', 'value.1': 'yellow',
                                                                         'value.2': "nan"}}},
                        'monotonicity': {-8723902965222060910: {'columns': [4, 5],
                                                                'constraints': {'direction': 'increasing'}}},
                        'max_difference': {-3381654230262078061: {'columns': [4],
                                                                  'constraints': {'max_difference': 10.0}}}}

    classes = json_to_class_parse(dict_constraints)

    assert isinstance(classes[0], ValueRange)
    assert classes[0].columns == [0, 2, 1]
    assert isinstance(classes[1], ValueRange)
    assert classes[1].columns == [3]
    assert isinstance(classes[2], ValueMonotonicity)
    assert classes[2].columns == [4, 5]
    assert isinstance(classes[3], ValueNominal)
    assert classes[3].columns == [3]
    assert isinstance(classes[4], ValueNominal)
    assert classes[4].columns == [7]
    assert isinstance(classes[5], ValueMaxDiff)
    assert classes[5].columns == [4]
