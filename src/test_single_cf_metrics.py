from metrics.single_cf_metrics import *

print('Test validity')
print(validity(0, 0))
print(validity(1, 0))

print('Test actionability')
print(actionability(np.array([1,1,0]), np.array([1,1,1]), np.array([0,0,1])))
print(actionability(np.array([1,0,0]), np.array([1,1,1]), np.array([0,0,1])))

print('Test proximity')
print(proximity(np.array([1,1,0]), np.array([1,1,1])))
print(proximity(np.array([1,1,0]), np.array([1,5,5])))

print('Test features changed')
print(features_changed(np.array([1,1,0]), np.array([1,1,1])))
print(features_changed(np.array([1,1,1]), np.array([1,1,1])))

print('Test feasibility')
print(feasibility(
    np.array([1,1,0]),
    np.array([1,1,1]),
    np.array([
        np.array([0,0,0]),
        np.array([1,0,0]),
        np.array([1,1,0])
    ])
))

print('Test discriminative power')
print(discriminative_power(
    np.array([1,1,0,0]),
    0,
    np.array([0,1,1,1]),
    1,
    np.array([
        np.array([0,0,0,0]),
        np.array([1,0,0,0]),
        np.array([1,1,0,1]),
        np.array([0,1,1,0]),
        np.array([0,0,1,1]),
        np.array([0,1,0,1]),
    ]),
    np.array([0, 0, 0, 1, 1, 1]),
    k = 4
))

print('Test preference dcg score')
print(preference_dcg_score(
    np.array([0, 3, 5, 6]),
    np.array([1, 4, 5, 6]),
    np.array([0, 2, 1]),
    k = 3
))

print('Test preference precision score')
print(preference_precision_score(
    np.array([0, 3, 5, 6]),
    np.array([1, 4, 5, 6]),
    np.array([0, 2, 1]),
    k = 3
))
