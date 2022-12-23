from typing import List, Any

from ..constraints import ValueRange, ValueMonotonicity, ValueNominal, ValueMaxDiff
import pandas as pd


def read_from_excel(path: str):
    ss = pd.read_excel(path, None)
    constraints: Any = {}
    for sheet in ss:
        ss[sheet]['hash'] = pd.Series(dtype=int)
        constraints[sheet] = {}
        for record_id in range(ss[sheet].shape[0]):
            # store current record
            r = ss[sheet].iloc[record_id]
            # store current column
            col_no = int(r[0])
            # store current constraints
            curr_con = dict(r)
            del curr_con['hash']
            del curr_con['column']
            # store current hash value
            h = hash(tuple(r[1:]))
            # group
            if h not in constraints[sheet].keys():
                constraints[sheet][h] = {'columns': [col_no],
                                         'constraints': curr_con}
            else:
                constraints[sheet][h]['columns'].append(col_no)
    return constraints


def json_to_class_parse(constraints):
    result: List[Any] = []
    for c in constraints['range'].values():
        result.append(ValueRange(columns=c['columns'], **c['constraints']))
    for c in constraints['monotonicity'].values():
        result.append(ValueMonotonicity(columns=c['columns'], **c['constraints']))
    for c in constraints['nominal'].values():
        result.append(ValueNominal(**c))
    for c in constraints['max_difference'].values():
        result.append(ValueMaxDiff(columns=c['columns'], **c['constraints']))
    return result
