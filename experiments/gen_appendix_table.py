import pandas as pd


data = pd.read_csv('experiments/results_visualization/cfs.csv')
data['L.p.'] = data.index + 1

data = data[['L.p.', 'explainer'] + data.drop(columns=['L.p.', 'explainer']).columns.tolist()]
data = data.rename(columns={'native.country': 'nat.cntry'})

print(data)

invalid = [30, 33, 35, 37, 40]
nonactionable = [42, 43, 44, 45, 46, 47, 48, 49, 50, 72, 73, 75, 76, 77, 78, 79, 80, 81]
pareto = [2, 4, 7, 8, 11, 13, 21, 22, 25, 26, 27, 38, 74]
rest = [0, 1, 3, 5, 6, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 23, 24, 28, 29, 31, 32, 34, 36, 39, 41, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71]


latex = data.to_latex(index=False)

def getcolorforrow(row):
    try:
        x = int(row.split('&')[0]) - 1
        if x in invalid: return 'invalid'
        if x in nonactionable: return 'nonactionable'
        if x in pareto: return 'pareto'
        if x in rest: return 'rest'
        else: print(x)
    except:
        return 'black'


result = ''
for i, row in enumerate(latex.split(r'\\')):
    if i > 1 and len(row) > 50:
        a = r'\rowcolor' + '{' + getcolorforrow(row) + '}' + row 
        # a = row
    else: 
        a = row
    result += r'  \\  ' + a
    
    
result = result.replace(r'\bottomrule', r'\hline')
result = result.replace(r'\toprule', r'\hline')
result = result.replace(r'\midrule', r'\hline')
    
print(result)