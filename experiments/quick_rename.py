from experiment_utils import load_data


def rename_explainers(dfs, _all):
    _all['explainer'] = _all['explainer'].str.replace('dice', 'dice-1')
    _all['explainer'] = _all['explainer'].str.replace('wachter', 'wachter-1')
    _all['explainer'] = _all['explainer'].str.replace('cfproto', 'cfproto-1')
    new_dfs = []
    for df in dfs:
        df['explainer'] = df['explainer'].str.replace('dice', 'dice-1')
        df['explainer'] = df['explainer'].str.replace('wachter', 'wachter-1')
        df['explainer'] = df['explainer'].str.replace('cfproto', 'cfproto-1')
        new_dfs.append(df)
    return new_dfs, _all

# dates = ['2023-03-22', '2023-03-23']
# dfs, _all, idcs = load_data('counterfactuals', dates, 'german')
# _all['explainer'] = _all['explainer'].str.replace('dice', 'dice-1')
# _all['explainer'] = _all['explainer'].str.replace('wachter', 'wachter-1')
# _all['explainer'] = _all['explainer'].str.replace('cfproto', 'cfproto-1')
# new_dfs = []
# for df in dfs:
#     df['explainer'] = df['explainer'].str.replace('dice', 'dice-1')
#     df['explainer'] = df['explainer'].str.replace('wachter', 'wachter-1')
#     df['explainer'] = df['explainer'].str.replace('cfproto', 'cfproto-1')
#     new_dfs.append(df)
# print(new_dfs)