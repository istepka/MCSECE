from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

SYMBOL_MAP={
        #'ideal_point_eucli': 'circle',
        'ideal_point_manha': 'circle-x',
        'cadex': 'square',
        'face': 'cross',
        'wachter': 'star',
        'fimap': 'pentagon',
        #'cem': 'triangle-up',
        #'cfproto': 'diamond-x',
        #'dice': 'star-square'
        }
COLOR_MAP={
        'ideal_point_eucli': '#ff7f00',
        'ideal_point_manha': '#1f78b4',
        'cadex': '#b15928',
        'face': '#33a02c',
        'wachter': '#fb9a99',
        'fimap': '#e31a1c',
        'cem': '#a6cee3',
        'cfproto': '#6a3d9a',
        'dice': '#b2df8a',
        }

def plot_tenary_visualization(results: List,
                              dataset_name: str,
                              write_path: str,
                              only_valid: bool = True
                              ) -> None:
    '''
    Plot tenary visualization for the results of the experiment 2.
    
    `results`: List of tuples (x, y, z, explainer, percentage)
    `dataset_name`: Name of the dataset
    `write_path`: Path to save the plot
    `only_valid`: Information for the title of the plot
    '''
    x,y,z,explainer,percentage = zip(*results)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    percentage = np.array(percentage)

    df = pd.DataFrame({
        'DiscriminativePower': x,
        'Feasibility': y,
        'Proximity': z,
        'explainer': explainer,
        'percentage': percentage,
    }).sort_values(by='explainer')

    fig = px.scatter_ternary(df, 
                            a="DiscriminativePower", 
                            b="Feasibility", 
                            c="Proximity", 
                            symbol="explainer", 
                            color="explainer",
                            #size="percentage", 
                            hover_name="explainer",  
                            # Change plot size
                            height=800,
                            width=1000,
                            symbol_map=SYMBOL_MAP,
                            color_discrete_map=COLOR_MAP,
                            size_max=20,
                            size=[20 for _ in range(len(x))]
                            )
    
    #fig.update_traces(fill='toself', selector=dict(type='scatterternary'))
    fig.update_traces(cliponaxis=False, selector=dict(type='scatterternary'), overwrite=True)
    fig.update_layout(showlegend=False, overwrite=True)
    fig.update_ternaries(
        aaxis_title="Discriminative Power",
        aaxis_layer= "below traces",
        aaxis_showline=True,
        aaxis_showgrid=True,
        aaxis_gridcolor="grey",
        aaxis_linecolor="black",
        
        baxis_title="Feasibility",
        baxis_layer= "below traces",
        baxis_showline=True,
        baxis_showgrid=True,
        baxis_gridcolor="grey",
        baxis_linecolor="black",
        
        caxis_title="Proximity",
        caxis_layer= "below traces",
        caxis_showline=True,
        caxis_showgrid=True,
        caxis_gridcolor="grey",
        caxis_linecolor="black",
        
        bgcolor="white",
        overwrite=True,
    )
    
    # Save plot to png
    if only_valid:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.eps"), format='eps')
    else:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}.eps"), format='eps')
        
    fig.show()
    
def generate_one_plot_for_all4():
    import matplotlib.pyplot as plt
    # trick to combine 4 plots in one
    # use plt.subplot to create 4 subplots
    fig, ax = plt.subplots(2, 2, figsize=(22, 20))
    ax = ax.flatten()
    # imread all 4 images
    for i, dataset in enumerate(['adult', 'german', 'compas', 'fico']):
        img = plt.imread(f'experiments\\tmp_results\\experiment2_{dataset}_valid.eps', format='eps')
        ax[i].imshow(img)
        # remove ticks
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        # remove line around the image
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
    
    # legend_img = plt.imread(f'experiments\\tmp_results\\legend.png')
    # ax[1].imshow(legend_img, extent=[0, 1, 0, 1])
    plt.savefig(f'experiments\\tmp_results\\4plots.eps', format='eps')
    plt.tight_layout()
    plt.show()
    
def generate_only_plotly_legend():
    l = []
    for symbol in SYMBOL_MAP.keys():
        l.append([0, 0, 1, symbol, 10])
    
    _df = pd.DataFrame(l, columns=['a', 'b', 'c', 'explainer', 'size'])
        
    fig = px.scatter_ternary(
        _df,
        a='a',
        b='b',
        c='c',
        symbol='explainer',
        color='explainer',
        symbol_map=SYMBOL_MAP,
        color_discrete_map=COLOR_MAP,
        # increase symbol size
        size='size',
        size_max=30
    )
    fig.update_layout(legend=dict(
                            title_font_family="Times New Roman",
                            font=dict(size=10)
                            
                            )   
                        )
    fig.write_image(f'experiments\\tmp_results\\legend_raw.png')
    fig.show()
    
    
if __name__ == '__main__':
    # Plot example
    results = (
        [1.0, 0.0, 0.0, 'ideal_point_eucli', 0.8],
        [0.2, 0.2, 0.3, 'cadex', 0.9],
        [0.7, 0.1, 0.5, 'face', 0.7],
        )
    
    #plot_tenary_visualization(results, 'test', 'experiments\\tmp_results\\')
    generate_one_plot_for_all4()
    #generate_only_plotly_legend()