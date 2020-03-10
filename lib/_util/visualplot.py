import lib._util.fileproc as fp

# Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)

import numpy as np

def faststat(df):
    na_count   = df.isna().sum()
    na_percent = na_count / len(df) * 100
    
    stats = [f'{x[0][0] :<30} | {str(x[0][1]) :<15} | {x[1][1] :<10,.5f} | {x[2][1] :,}'
             for x in zip(df.dtypes.items(), na_percent.items(), na_count.items())]
    
    print(df.shape)
    print('Column                         | Type            | N/A %      | N/A Count')
    print('-------------------------------------------------------------------------')
    print('\n'.join(stats))

# TODO - change to use plotly theme
def figure(data, title=None, xlabel=None, ylabel=None, hovermode='x', showlegend=True):
    axis_dict = dict(
        title=xlabel,
        gridcolor='rgb(159, 197, 232)'
    )
    xaxis_kwargs = {f'xaxis{x+1 if x != 0 else ""}': axis_dict
                    for x in range(50)}
    axis_dict['title'] = ylabel
    yaxis_kwargs = {f'yaxis{x+1 if x != 0 else ""}': axis_dict
                    for x in range(50)}

    layout = go.Layout(
        **xaxis_kwargs,
        **yaxis_kwargs,
        title = title,
        hovermode = hovermode,
        showlegend = showlegend,
        legend_orientation = 'h',
        plot_bgcolor = 'rgba(0, 0, 0, 0)'
    )

    return go.Figure(data=data, layout=layout)

def update_axis(fig, axis_count, gridcolor='rgb(159, 197, 232)'):
    for x in range(axis_count):
        suffix = x+1 if x != 0 else ''
        fig['layout'][f'xaxis{suffix}']['gridcolor'] = gridcolor
        fig['layout'][f'yaxis{suffix}']['gridcolor'] = gridcolor

def plot_graph(data, title, xlabel=None, ylabel=None, generate_file=True, out_path=None,
               layout_width=None, layout_height=None, hovermode='x', showlegend=True):
    fig = figure(data, title, xlabel, ylabel, hovermode=hovermode, showlegend=showlegend)
    fig.update_layout(width=layout_width, height=layout_height)

    if generate_file:
        generate_plot(fig, out_path, title)
    else:
        generate_plot(fig)

def generate_plot(fig, out_path=None, out_filename=None, axis_count=None):
    if axis_count is not None:
        update_axis(fig, axis_count)

    if out_path is None or out_filename is None:
        iplot(fig)
    else:
        fp.create_directory(out_path)
        out_filename = f'{out_filename}.html' if '.html' not in out_filename else out_filename
        out_file     = f'{out_path}{out_filename}'
        plot(fig, filename=out_file, auto_open=False)
        
        print(f'Generated: {out_file}')

def plot_subplots(data, max_col, title, subplot_titles=None, out_path=None, showlegend=False, layout_width=None, layout_height=None):
    max_row = int(np.ceil(len(data) / max_col))
    
    fig = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles, x_title=title)

    for index, trace in enumerate(data):
        col = index +1

        if col <= max_col:
            row = 1
        else:
            quotient = int(col / max_col)
            col -= (max_col * quotient)
            if col == 0:
                col = max_col
            elif col == 1:
                row += 1

        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(showlegend=showlegend, width=layout_width, height=layout_height, plot_bgcolor='rgba(0, 0, 0, 0)')
    generate_plot(fig, out_path=out_path, out_filename=title, axis_count=len(data))

def datagroups_subplots(data_groups, max_col, title, subplot_titles=None, out_path=None, showlegend=False, layout_width=None, layout_height=None):
    max_row = int(np.ceil(len(data_groups) / max_col))
    
    fig = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles, x_title=title)
    for index, data_group in enumerate(data_groups):
        col = index +1

        if col <= max_col:
            row = 1
        else:
            quotient = int(col / max_col)
            col -= (max_col * quotient)
            if col == 0:
                col = max_col
            elif col == 1:
                row += 1

        for data in data_group:
            fig.add_trace(data, row=row, col=col)

    fig.update_layout(showlegend=showlegend, width=layout_width, height=layout_height, plot_bgcolor='rgba(0, 0, 0, 0)', barmode='overlay')
    generate_plot(fig, out_path=out_path, out_filename=title, axis_count=len(data_groups))

def scatter(df, x_col, y_col, category_col=None, title='Scatter', out_path=None, layout_width=None, layout_height=None):
    data = []
    if category_col is None:
        data.append(go.Scattergl(
            x = df[x_col],
            y = df[y_col],
            mode = 'markers',
            marker = {
                'opacity': .5
            }
        ))
    else:
        for category in sorted(df[category_col].unique()):
            tmp_df = df[df[category_col] == category]
            data.append(go.Scattergl(
                x = tmp_df[x_col],
                y = tmp_df[y_col],
                mode = 'markers',
                marker = {
                    'opacity': .5
                },
                name = str(category)
            ))
    plot_graph(data, title=title, out_path=out_path, layout_width=layout_width, layout_height=layout_height, hovermode=None)
    
def bar(df, x_col, y_cols, title='Bar', out_path=None, layout_width=None, layout_height=None):
    data = []
    
    for y_col in y_cols:
        data.append(go.Bar(
            x = df[x_col],
            y = df[y_col]
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in y_cols]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def histogram(df, title='Histogram', out_path=None, layout_width=None, layout_height=None):
    data = []

    for column in df.columns:
        data.append(go.Histogram(
            x = df[column]
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in df.columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def boxplot(df, x_col=None, title='Box-Plot', out_path=None, layout_width=None, layout_height=None, numeric_only=True):
    data = []

    columns = [k for k,v in df.dtypes.items() if 'float' in str(v) or 'int' in str(v)] if numeric_only else df.columns
    columns = [x for x in columns if x != x_col]
    for column in columns:
        data.append(go.Box(
            x = df[x_col],
            y = df[column],
            boxpoints = 'outliers', # all, outliers, suspectedoutliers
            boxmean = True
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def violinplot(df, x_col=None, title='Violin-Plot', out_path=None, layout_width=None, layout_height=None, numeric_only=True):
    data = []

    columns = [k for k,v in df.dtypes.items() if 'float' in str(v) or 'int' in str(v)] if numeric_only else df.columns
    columns = [x for x in columns if x != x_col]
    for column in columns:
        data.append(go.Violin(
            x = df[x_col],
            y = df[column],
            box_visible = True,
            meanline_visible = True,
            points = 'outliers' # all, outliers, suspectedoutliers
        ))

    max_col = 2
    subplot_titles = [f'{x.lower()}' for x in columns]
    plot_subplots(data, max_col, title, subplot_titles=subplot_titles, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def corrmatrix(df, title='Correlation Matrix', out_path=None, layout_width=None, layout_height=None):
    corrmat = df.corr()
    corrmat = corrmat.where(np.tril(np.ones(corrmat.shape), k=-1).astype(np.bool))

    data = go.Heatmap(
        x = corrmat.index,
        y = corrmat.columns,
        z = corrmat.values,
        colorscale = 'RdBu'
    )
    plot_graph(data, title=title, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def heatmap(df, x_col, y_col, z_col, colorscale='RdBu', text=None, hoverinfo='all',
            title='Heatmap', out_path=None, layout_width=None, layout_height=None):
    data = go.Heatmap(
        x = df[x_col],
        y = df[y_col],
        z = df[z_col],
        colorscale = colorscale,
        hoverinfo = hoverinfo,
        text = text
    )
    plot_graph(data, title=title, out_path=out_path, layout_width=layout_width, layout_height=layout_height)

def pairplot(df, title='Pair-Plot', out_path=None, layout_width=None, layout_height=None, numeric_only=True, category=None):
    columns = [k for k,v in df.dtypes.items() if 'float' in str(v) or 'int' in str(v)] if numeric_only else df.columns

    fig = px.scatter_matrix(df, dimensions=columns, color=category, title=title, opacity=.5)

    fig.update_layout(width=layout_width, height=layout_height)
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    generate_plot(fig, out_path=out_path, out_filename=title)