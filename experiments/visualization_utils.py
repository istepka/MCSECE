from typing import List
import numpy as np
import pandas as pd
import plotly.express as px
import os


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
        'DiscriminativePower(9)': x,
        'K_Feasibility(3)': y,
        'Proximity': z,
        'explainer': explainer,
        'percentage': percentage,
    }).sort_values(by='explainer')

    fig = px.scatter_ternary(df, 
                            a="DiscriminativePower(9)", 
                            b="K_Feasibility(3)", 
                            c="Proximity", 
                            color="explainer", 
                            size="percentage", 
                            size_max=18, 
                            hover_name="explainer", 
                            color_continuous_scale=px.colors.sequential.Plasma,
                            # Title
                            title=f"Best Explainer for {dataset_name} dataset" + (" (only valid)" if only_valid else ""),
                            # Show axes arrow    
                            # Change plot size
                            height=800,
                            width=1000,
                            )
    # Save plot to png
    if only_valid:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}_valid.png"))
    else:
        fig.write_image(os.path.join(write_path, f"experiment2_{dataset_name}.png"))
        
    fig.show()
