import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns

def fewpoints(x,y,z,RSO,download):

  columns=['x','y','z']+list(RSO.keys())
  length=np.size(RSO[list(RSO.keys())[0]])
  DF=pd.DataFrame(data=np.zeros([length,len(columns)]),columns=columns)
  counter=0
  for xx in range(len(x)):
    for yy in range(len(y)):
      for zz in range(len(z)):
        DF.loc[counter,'x']=x[xx]
        DF.loc[counter,'y']=y[yy]
        DF.loc[counter,'z']=z[zz]
        for col in columns[3:]:
          mult=1
          if col[:3]=='eps':
            mult=10**6
          DF.loc[counter,col]=RSO[col][yy,xx,zz]*mult
        counter=counter+1
  if download:
    from google.colab import files
    DF.to_excel("PLEA.xlsx")
    files.download('PLEA.xlsx')
  return DF
  
def plot_interactive_heatmap(title, data, label, x, z, H, aspect=(15,8), interpolate=True):
    import plotly.graph_objects as go
    """ Plots interactive heatmap, adjusted for exact edge alignment. """
    # Verify input data shape matches coordinate lengths
    if data.shape != (len(x), len(z)):
        raise ValueError(f"Input data shape {data.shape} does not match len(x)={len(x)}, len(z)={len(z)}. Expected shape ({len(x)}, {len(z)}).")

    if interpolate:
        from scipy.interpolate import RegularGridInterpolator
        # Create finer grids for interpolation
        x_new = np.linspace(x.min(), x.max(), 300)
        z_new = np.linspace(z.min(), z.max(), 300)
        # ... (interpolation logic remains the same) ...
        interpolator = RegularGridInterpolator((x, z), data, method='linear', bounds_error=False, fill_value=None)
        X_new_grid, Z_new_grid = np.meshgrid(x_new, z_new, indexing='xy')
        points_to_interpolate = np.vstack((X_new_grid.ravel(), Z_new_grid.ravel())).T
        data_plot = interpolator(points_to_interpolate).reshape(X_new_grid.shape)
        # --- Coordinates used for plotting ---
        plot_x_coords = x_new
        plot_z_coords = z_new
        if np.isnan(data_plot).any() or np.isinf(data_plot).any():
            print("Warning: Interpolated data contains NaNs or infinite values.")
    else:
        # Use original data without interpolation
        x_new, z_new = x, z # Keep original names for consistency downstream if needed
        data_plot = data.T
        # --- Coordinates used for plotting ---
        plot_x_coords = x
        plot_z_coords = z

    # --- Calculate x-axis range to encompass cell edges ---
    if len(plot_x_coords) > 1:
        # Calculate half the distance between the first two points and last two points
        half_dx_first = (plot_x_coords[1] - plot_x_coords[0]) / 2.0
        half_dx_last = (plot_x_coords[-1] - plot_x_coords[-2]) / 2.0
        # Define range from the left edge of the first cell to the right edge of the last cell
        x_range_min_edge = plot_x_coords[0] - half_dx_first
        x_range_max_edge = plot_x_coords[-1] + half_dx_last
    elif len(plot_x_coords) == 1:
        # Handle single point case (arbitrary width, or use original x bounds?)
        x_range_min_edge = plot_x_coords[0] - 0.5 # Default width of 1 centered on the point
        x_range_max_edge = plot_x_coords[0] + 0.5
    else:
        # No points, fallback to original x min/max
        x_range_min_edge = x.min()
        x_range_max_edge = x.max()

    # --- Plotting using Plotly ---
    fig = go.Figure(data=go.Heatmap(
        z=data_plot,
        x=np.round(plot_x_coords, 3), # Use the potentially interpolated coords
        y=np.round(plot_z_coords, 3), # Use the potentially interpolated coords
        colorscale='RdBu',
        colorbar=dict(title=label),
        reversescale=True,
        zmid=0,
        connectgaps=False
    ))

    # Add horizontal lines for layer boundaries (using plot_x_coords bounds)
    if H is not None and len(H) > 0:
        H_np = np.asarray(H)
        cumulative_sums = np.cumsum(H_np)
        plot_z_max = plot_z_coords.max()
        # Use the calculated edge range for lines to span the full heatmap width
        line_x_min, line_x_max = x_range_min_edge, x_range_max_edge
        for h in cumulative_sums:
            if h <= plot_z_max:
                 fig.add_trace(go.Scatter(x=[line_x_min, line_x_max], y=[h, h], mode='lines',
                                          line=dict(color='black', width=1, dash='dash'),
                                          opacity=0.7,
                                          showlegend=False, hoverinfo='none'))

    # --- Update layout ---
    fig.update_layout(
        title=title,
        xaxis_title='x Coordinate',
        yaxis_title='z Coordinate (Depth)',
        yaxis_autorange='reversed',
        width=int(aspect[0] * 80),
        height=int(aspect[1] * 80),

        # --- Set the calculated x-axis range ---
        xaxis=dict(
            range=[x_range_min_edge, x_range_max_edge],
            # Optional: Removing scaleanchor/scaleratio might give more direct range control
            # scaleanchor="y",
            # scaleratio=1
            # Constrain axis allows zooming/panning only within the set range
            constrain='domain',
        ),
        yaxis=dict(
             # Setting constrain here too ensures z-axis doesn't pan beyond reversed range
             constrain='domain',
        )
        # Optional: Minimize margin padding further if needed
        # margin=dict(l=5, r=5, t=30, b=5, pad=0), # Adjust l,r,t,b as needed
    )

    fig.show()