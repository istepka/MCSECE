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
        #'dice-1': 'triangle-up',
        'cfproto-1': 'diamond-x',
        'cem': 'x'
        }
COLOR_MAP={
        #'ideal_point_eucli': '#ff7f00',
        'ideal_point_manha': '#377eb8',
        'cadex': '#e41a1c',
        'face': '#4daf4a',
        'wachter': '#984ea3',
        'fimap': '#ff7f00',
        'dice-1': '#e6ab02',
        'cfproto-1': '#a65628',
        'cem': '#f781bf',
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
                            height=1000,
                            width=1500,
                            symbol_map=SYMBOL_MAP,
                            color_discrete_map=COLOR_MAP,
                            size_max=20,
                            size=[20 for _ in range(len(x))],
                            opacity=1,
                            )
    
    #fig.update_traces(fill='toself', selector=dict(type='scatterternary'))
    fig.update_traces(cliponaxis=False, selector=dict(type='scatterternary'), overwrite=True)
    fig.update_layout(showlegend=False, overwrite=True)
    fig.update_ternaries(
        aaxis_title="<b>Discriminative Power</b>",
        aaxis_title_font_size=25,
        aaxis_layer= "below traces",
        aaxis_showline=True,
        aaxis_showgrid=True,
        aaxis_gridcolor="grey",
        aaxis_linecolor="black",
        # turn off ticks
        aaxis_tickfont=dict(color="white"),
        
        baxis_title="<b>Feasibility</b>",
        baxis_title_font_size=25,
        baxis_layer= "below traces",
        baxis_showline=True,
        baxis_showgrid=True,
        baxis_gridcolor="grey",
        baxis_linecolor="black",
        baxis_tickfont=dict(color="white"),
        
        caxis_title="<b>Proximity</b>",
        caxis_title_font_size=25,
        caxis_layer= "below traces",
        caxis_showline=True,
        caxis_showgrid=True,
        caxis_gridcolor="grey",
        caxis_linecolor="black",
        caxis_tickfont=dict(color="white"),
        
        bgcolor="white",
        overwrite=True,
    )
    # fig.update_layout(legend=dict(
    #                     title_font_family="Times New Roman",
    #                     font=dict(
    #                         size=35,
    #                         ),
    #                     itemsizing='constant',
    #                     ), 
    #                 overwrite=True) 
    # Save plot to png
    if only_valid:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.svg"), format='svg')
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.png"), format='png')
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.pdf"), format='pdf')
    else:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}.svg"), format='svg')
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}.png"), format='png')
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.pdf"), format='pdf')
        
    fig.show()
    
def generate_one_plot_for_all4():
    import matplotlib.pyplot as plt
    # trick to combine 4 plots in one
    # use plt.subplot to create 4 subplots
    fig, ax = plt.subplots(2, 2, figsize=(22, 20))
    ax = ax.flatten()
    # imread all 4 images
    for i, dataset in enumerate(['adult', 'german', 'compas', 'fico']):
        img = plt.imread(f'experiments\\tmp_results\\experiment2_{dataset}_valid.png', format='png')
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
    plt.savefig(f'experiments\\tmp_results\\4plots.svg', format='svg', dpi=300)
    plt.savefig(f'experiments\\tmp_results\\4plots.png', format='png', dpi=300)
    plt.tight_layout()
    plt.show()
    
def generate_only_plotly_legend():
    _SYMBOL_MAP={
        #'ideal_point_eucli': 'circle',
        'our ensemble (Manhattan)': 'circle-x',
        'Cadex': 'square',
        'FACE': 'cross',
        'Wachter': 'star',
        'Fimap': 'pentagon',
        #'dice-1': 'triangle-up',
        'CFProto': 'diamond-x',
        'CEM': 'x'
        }
    _COLOR_MAP={
            #'ideal_point_eucli': '#ff7f00',
            'our ensemble (Manhattan)': '#377eb8',
            'Cadex': '#e41a1c',
            'FACE': '#4daf4a',
            'Wachter': '#984ea3',
            'Fimap': '#ff7f00',
            'Dice': '#e6ab02',
            'CFProto': '#a65628',
            'CEM': '#f781bf',
            }
    l = []
    for symbol in _SYMBOL_MAP.keys():
        l.append([0.1, 0.4, 0.5, symbol, 25])
    
    _df = pd.DataFrame(l, columns=['a', 'b', 'c', 'explainer', 'size'])
        
    fig = px.scatter_ternary(
        _df,
        a='a',
        b='b',
        c='c',
        symbol='explainer',
        color='explainer',
        symbol_map=_SYMBOL_MAP,
        color_discrete_map=_COLOR_MAP,
        # increase symbol size
        size='size',
        size_max=25,
        opacity=1,
        #marker_size=50,
    )
    fig.update_layout(legend=dict(
                            title_font_family="Times New Roman",
                            font=dict(size=35),
                            itemsizing='trace',
                            orientation='h',
                            ), 
                      overwrite=True) 
    fig.write_image(f'experiments\\tmp_results\\legend_raw.png', width=2000, height=1200)
    fig.write_image(f'experiments\\tmp_results\\legend_raw.svg', width=2000, height=1200, format='svg')
    fig.show()
    
    
if __name__ == '__main__':
    # Plot example
    results = (
        [1.0, 0.0, 0.0, 'ideal_point_eucli', 0.8],
        [0.2, 0.2, 0.3, 'cadex', 0.9],
        [0.7, 0.1, 0.5, 'face', 0.7],
        )
    
    #plot_tenary_visualization(results, 'test', 'experiments\\tmp_results\\')
    #generate_one_plot_for_all4()
    generate_only_plotly_legend()
    
    # from cairosvg import svg2png
    # svg2png(url="experiments\\tmp_results\\experiment2_fico_valid.svg",  write_to="experiments\\tmp_results\\experiment2_fico_valid.png")