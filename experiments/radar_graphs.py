import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from experiment_utils import load_data
import pandas as pd
import os
 
value = [90, 85, 85, 80, 75, 65, 40, 40, 10]
names = ['SQL', 'uczenie nadzorowane', 'zaawansowana analityka', 'rozumienie biznesu',
         'storytelling', 'python', 'głębokie uczenie', 'NLP', 'uczenie nienadzorowane']

DATES = ['2023-03-12', '2023-03-14', '2023-03-15'] 
DIR = os.path.join(os.getcwd(), 'experiments', 'results', 'radar_graphs')
DATA_DIR = os.path.join(os.getcwd(), 'experiments', 'tmp_results3')
if os.path.exists(DIR) is False:
    os.makedirs(DIR)

whitelist_explainers = [
    'Dice',
    'Fimap',
    'FACE',
    'Cadex',
    'Wachter',
    'CFProto',
    'CEM',
    'GrowingSpheres',
    'ActionableRecourse',
    'our approach (Manhattan)',
    'our approach (Euclidean)',
    'our approach (Chebyshev)',
]

for dataset in ['adult', 'german', 'compas', 'fico']:
    
    valid_scores_df_all = pd.read_csv(os.path.join(DATA_DIR, f'experiment1_{dataset}.csv_valid.csv'), index_col=0)
    print(valid_scores_df_all.head(5))
    data = []
    
    rows = 4
    cols = 3
    subplots = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'polar'} for i in range(cols)] for _ in range(rows)])
    
    valid_scores_df_all = valid_scores_df_all.rename(
        columns={
            'discriminative_power_9': 'DiscriminativePower',
            'k_feasibility_3': '(-)Feasibility',
            'proximity': '(-)Proximity',
            'sparsity': '(-)Sparsity',
            'instability': '(-)Instability',
        },
        index={
            'dice': 'Dice',
            'fimap': 'Fimap',
            'face': 'FACE',
            'cadex': 'Cadex',
            'wachter': 'Wachter',
            'cfproto': 'CFProto',
            'cem': 'CEM',
            'growing-spheres': 'GrowingSpheres',
            'actionable-recourse': 'ActionableRecourse',
            'ideal_point_manha': 'our approach (Manhattan)',
            'ideal_point_eucli': 'our approach (Euclidean)',
            'ideal_point_cheby': 'our approach (Chebyshev)',
        }
    )
    # Min max normalization
    columns = valid_scores_df_all.columns.tolist()
    columns.remove('actionable')
    columns.remove('coverage')
    
    min_cols = list(set(columns) - set(['DiscriminativePower']))
    valid_scores_df_all['DiscriminativePower'] = valid_scores_df_all['DiscriminativePower'] * 100
    
    valid_scores_df_all[min_cols] = 100 * (valid_scores_df_all[min_cols] - 0) \
                                    / (valid_scores_df_all[min_cols].max(axis=0) - 0)
    # valid_scores_df_all[min_cols] = 100 * (valid_scores_df_all[min_cols] - valid_scores_df_all[min_cols].min(axis=0)) \
    #                                 / (valid_scores_df_all[min_cols].max(axis=0) - valid_scores_df_all[min_cols].min(axis=0))

    # Invert values for cost criteria
    valid_scores_df_all[min_cols] = 100 - valid_scores_df_all[min_cols] 
    
    for explainer in valid_scores_df_all.index.tolist():
        
        if explainer not in whitelist_explainers:
            continue
        

        value = valid_scores_df_all.loc[explainer][columns]
        print(value.shape)
        value = value.to_list()
        
        data.append(go.Scatterpolar(
                r=value,
                theta=columns,
                fill='toself',
                name=explainer,
        ))
    
    for i in range(len(data)):
        subplots.add_trace(data[i], row=i // cols + 1, col=i % cols + 1)
        
    subplots.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, 
            range=[0,100], 
            tickmode='array', 
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0', '25', '50', '75', '100'],
            tickangle=0,
            )),
        showlegend=True,
        legend=dict(orientation="h"),
    )


    # Save plot to pdf
    pio.write_image(subplots, os.path.join(DIR, f'radar_{dataset}.svg'), format='svg', width=1200, height=1500, scale=1)
    pio.write_image(subplots, os.path.join(DIR, f'radar_{dataset}.png'), format='png', width=1200, height=1500, scale=1)
    #subplots.show()
    
    
    
    
    # Plot ideal point vs face vs cadex 
    fig = go.Figure()
    for explainer in ['FACE', 'Cadex', 'our approach (Manhattan)']:
        fig.add_trace(go.Scatterpolar(
            r=valid_scores_df_all.loc[explainer][columns],
            theta=columns,
            fill='toself',
            name=explainer,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True,
            range=[0, 100],
            tickmode='array',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0', '25', '50', '75', '100'],
            tickangle=0,
            )),
        showlegend=False,
        # Increase font size and boldness
        font=dict(
            size=35,
            color="black",
        ),
        legend=dict(font=dict(size=30),
                    itemsizing='trace',
                    orientation='h',
                    ),
    )
    
                       
    pio.write_image(fig, os.path.join(DIR, f'radar_{dataset}_mahattan_vs_face_vs_cadex.png'), format='png', width=1750, height=1400, scale=1)
    pio.write_image(fig, os.path.join(DIR, f'radar_{dataset}_mahattan_vs_face_vs_cadex.svg'), format='svg', width=1750, height=1400, scale=1)
        
        
    

    
        
    