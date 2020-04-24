import lib._util.fileproc as fp

# Plotly
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
# init_notebook_mode(connected=True)
pio.templates.default = 'seaborn'

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress, probplot
from scipy.cluster.hierarchy import linkage

def faststat(df, max_rows=None):
    # N/A info
    stat_df = df.isna().sum().to_frame(name='N/A Count')
    stat_df['N/A Ratio'] = np.round(stat_df['N/A Count'] / len(df), 5)

    # Type info
    stat_df = stat_df.merge(df.dtypes.to_frame(name='Type'), left_index=True, right_index=True, how='left')
    
    # Unique info
    stat_df['Uniq. Count'] = stat_df.index.to_series().apply(lambda x: -1)
    for x in stat_df.index:
        stat_df.at[x, 'Uniq. Count'] = len(df[x].unique())
    stat_df['Uniq. Ratio'] = np.round(stat_df['Uniq. Count'] / len(df), 5)

    default_max_rows = pd.options.display.max_rows
    pd.set_option('display.max_rows', max_rows)
    print(df.shape)
    print(stat_df)
    pd.set_option('display.max_rows', default_max_rows)

def value_count(df, column, max_rows=None):
    count_df = df[column].value_counts().to_frame(name='Count')
    ratio_df = df[column].value_counts(normalize=True).to_frame(name='Ratio')
    stat_df  = count_df.merge(ratio_df, left_index=True, right_index=True, how='left')
    stat_df.index.name = column
    
    default_max_rows = pd.options.display.max_rows
    pd.set_option('display.max_rows', max_rows)
    print(f'Unique: {len(stat_df) :,}')
    print(stat_df)
    pd.set_option('display.max_rows', default_max_rows)

def generate_plot(fig, out_path=None, out_filename=None, to_image=False):
    if out_path is None or out_filename is None:
        iplot(fig)
    else:
        fp.create_directory(out_path)
        out_filename = f'{out_filename}.html' if '.html' not in out_filename else out_filename
        out_file     = f'{out_path}{out_filename}'

        if to_image:
            out_file = out_file.replace('.html', '.png')
            fig.write_image(out_file)
        else:
            plot(fig, filename=out_file, auto_open=False)
        
        print(f'Generated: {out_file}')

def plot_subplots(data, max_col, title,
                  out_path=None, to_image=False,
                  layout_kwargs={}, xaxis_titles=[], yaxis_titles=[], subplot_titles=None):

    max_row = int(np.ceil(len(data) / max_col))
    fig     = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles)
    
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

        # Update axis label
        axisnum = '' if index == 0 else index +1
        try:
            fig['layout'][f'xaxis{axisnum}']['title'] = xaxis_titles[index]
        except IndexError:
            fig['layout'][f'xaxis{axisnum}']['title'] = trace['name']
            
        try:
            fig['layout'][f'yaxis{axisnum}']['title'] = yaxis_titles[index]
        except IndexError:
            pass
    
    layout_kwargs['title'] = title
    fig.update_layout(**layout_kwargs)
    
    generate_plot(fig, out_path=out_path, out_filename=title, to_image=to_image)

def datagroups_subplots(data_groups, max_col, title,
                        out_path=None, to_image=False,
                        layout_kwargs={}, xaxis_titles=[], yaxis_titles=[], subplot_titles=None):

    max_row = int(np.ceil(len(data_groups) / max_col))
    fig     = make_subplots(rows=max_row, cols=max_col, subplot_titles=subplot_titles)

    for index, data in enumerate(data_groups):
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

        for trace in data:
            fig.add_trace(trace, row=row, col=col)

        # Update axis label
        axisnum = '' if index == 0 else index +1
        try:
            fig['layout'][f'xaxis{axisnum}']['title'] = xaxis_titles[index]
        except IndexError:
            pass
            
        try:
            fig['layout'][f'yaxis{axisnum}']['title'] = yaxis_titles[index]
        except IndexError:
            pass
            
    layout_kwargs['title'] = title
    fig.update_layout(**layout_kwargs)

    generate_plot(fig, out_path=out_path, out_filename=title, to_image=to_image)



# BASIC
def histogram(df, title='Histogram',
              out_path=None, max_col=2, layout_kwargs={}, to_image=False,
              bin_algo='default', str_length=None, hidden_char='..'):

    bin_algos = ['default', 'count', 'width']
    assert bin_algo in bin_algos, f'bin_algo not in valid list: {bin_algos}'

    data    = []
    colors  = DEFAULT_PLOTLY_COLORS
    columns = df.columns
    
    for column in columns:
        try:
            # https://www.qimacros.com/histogram-excel/how-to-determine-histogram-bin-interval/
            n_data = len(np.unique(df[column]))
            n_bins = int(np.ceil(np.sqrt(n_data)))
            width  = (df[column].max() - df[column].min()) / n_bins

            nbinsx = n_bins if bin_algo == 'count' else None
            xbins  = {'size': width} if bin_algo == 'width' else None

        except TypeError:
            nbinsx = None
            xbins  = None

        x = df[column].copy()
        if x.dtype == object:
            x = x.sort_values()
            x = np.where(x.str.len() > str_length, x.str.slice(stop=str_length).str.strip() + hidden_char, x)

        data.append(go.Histogram(
            x=x,
            name=column,
            showlegend=False,
            nbinsx=nbinsx,
            xbins=xbins,
            marker={'color': colors[0]}
        ))
        
    plot_subplots(data, max_col=max_col, title=title, out_path=out_path,
                  layout_kwargs=layout_kwargs, to_image=to_image)

def kde(df, title='KDE', color=None,
        out_path=None, max_col=2, layout_kwargs={}, to_image=False):
    
    columns = df.select_dtypes(include='number')
    columns = [x for x in columns if x != color]

    data_groups = []
    groups      = [1] if color is None else np.unique(df[color])

    for column in columns:
        data   = []
        colors = DEFAULT_PLOTLY_COLORS

        for index, group in enumerate(groups):
            tmp_df = df if color is None else df[df[color] == group]
            kde    = sm.nonparametric.KDEUnivariate(tmp_df[column].astype('float'))
            kde.fit()
            
            data.append(go.Scattergl(
                x=kde.support,
                y=kde.density,
                name=column if color is None else f'{column}: {group}',
                marker={'color': colors[index % len(colors)]}
            ))
        data_groups.append(data)

    if color is None:
        layout_kwargs['showlegend'] = False

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=columns,
                        layout_kwargs=layout_kwargs, to_image=to_image)

def box(df, title='Box', color=None,
        out_path=None, max_col=2, layout_kwargs={}, to_image=False):

    columns = df.select_dtypes(include='number')
    columns = [x for x in columns if x != color]

    data_groups = []
    groups      = [1] if color is None else np.unique(df[color])

    for column in columns:
        data   = []
        colors = DEFAULT_PLOTLY_COLORS

        for index, group in enumerate(groups):
            data.append(go.Box(
                y=df[column] if color is None else df[df[color] == group][column],
                name=column if color is None else group,
                marker={'color': colors[index % len(colors)]}
            ))
        data_groups.append(data)

    if color is None:
        layout_kwargs['showlegend'] = False

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        yaxis_titles=[] if color is None else columns,
                        layout_kwargs=layout_kwargs, to_image=to_image)

def box_categorical(df, y, title='Box',
                    out_path=None, max_col=2, layout_kwargs={}, to_image=False):

    columns = df.select_dtypes(include='object')
    columns = [x for x in columns if x != y]

    data_groups = []
    for column in columns:
        median_df = df.groupby(column).agg(
            BOX_CATEGORICAL_median=(y, 'median')
        ).reset_index().sort_values(by='BOX_CATEGORICAL_median')

        tmp_df = df[[column, y]].copy()
        tmp_df = tmp_df.merge(median_df, on=column, how='left')
        tmp_df = tmp_df.sort_values(by='BOX_CATEGORICAL_median')

        fig = px.box(tmp_df, x=column, y=y)
        data_groups.append(fig['data'])

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=columns,
                        yaxis_titles=[y if i % max_col == 0 else None for i,_ in enumerate(columns)],
                        layout_kwargs=layout_kwargs, to_image=to_image)

def bar_mean_median(df, y, title='Bar - Mean-Median',
                    out_path=None, max_col=2, layout_kwargs={}, to_image=False):

    columns = df.select_dtypes(include='object')
    columns = [x for x in columns if x != y]

    data_groups  = []
    stats        = ['mean', 'median']
    mean, median = df[y].mean(), df[y].median()
    colors       = DEFAULT_PLOTLY_COLORS

    for column in columns:
        stat_df = df.groupby(column).agg(
            mean=(y, 'mean'),
            median=(y, 'median')
        ).reset_index()

        for stat in stats:
            data = []
            data.append(go.Bar(
                x=stat_df.sort_values(by=stat)[column],
                y=stat_df.sort_values(by=stat)[stat],
                showlegend=False,
                marker={'color': colors[0]}
            ))
            data.append(go.Scattergl(
                x=stat_df.sort_values(by=stat)[column],
                y=[mean if stat == 'mean' else median for x in range(len(stat_df))],
                showlegend=False,
                marker={'color': 'red'},
                mode='lines',
                line={'dash': 'dash'}
            ))
            data_groups.append(data)

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=[f'{x} - {y.title()}' for x in columns for y in stats],
                        layout_kwargs=layout_kwargs, to_image=to_image)

# Reference: https://stackoverflow.com/questions/51170553/qq-plot-using-plotly-in-python
def prob(df, title='Probability',
         out_path=None, max_col=2, layout_kwargs={}, to_image=False):
    
    columns     = df.select_dtypes(include='number')
    data_groups = []
    colors      = DEFAULT_PLOTLY_COLORS

    for column in columns:
        (osm, osr), (slope, intercept, r) = probplot(df[column])
        line_points = np.array([osm[0], osm[-1]])

        data = []
        data.append(go.Scattergl(
            x=osm,
            y=osr,
            mode='markers',
            showlegend=False,
            marker={'color': colors[0]}
        ))
        data.append(go.Scattergl(
            x=line_points,
            y=intercept + slope * line_points,
            mode='lines',
            showlegend=False,
            marker={'color': 'red'},
        ))
        data_groups.append(data)

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=['Theoretical Quantiles' for _ in columns],
                        yaxis_titles=['Ordered Values' if i % max_col == 0 else None for i,_ in enumerate(columns)],
                        subplot_titles=list(columns),
                        layout_kwargs=layout_kwargs, to_image=to_image)

def line(df, xy_tuples, title='Line',
         out_path=None, max_col=2, layout_kwargs={}, to_image=False,
         line_kwargs={}, scattergl=False):
    
    data_groups = []
    colors      = DEFAULT_PLOTLY_COLORS

    for index, (x, y) in enumerate(xy_tuples):
        if scattergl:
            data = []
            data.append(go.Scattergl(
                x=df[x].values,
                y=df[y].values,
                showlegend=False,
                marker={'color': colors[0]},
                **line_kwargs
            ))
            data_groups.append(data)

        else:
            fig = px.line(df, x=x, y=y, **line_kwargs)
            if index != 0:
                for data in fig['data']:
                    data['showlegend'] = False
            data_groups.append(fig['data'])

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=[xy[0] for xy in xy_tuples],
                        yaxis_titles=[xy[1] for xy in xy_tuples],
                        layout_kwargs=layout_kwargs, to_image=to_image)
    
def scatter(df, xy_tuples, title='Scatter', color=None,
            out_path=None, max_col=2, layout_kwargs={}, to_image=False,
            scatter_kwargs={}):

    data_groups = []
    r_values    = []
    p_values    = []
    trendline   = False if scatter_kwargs.get('trendline') is None else True

    for index, (x, y) in enumerate(xy_tuples):
        fig = px.scatter(df, x=x, y=y, color=color, **scatter_kwargs)

        if trendline:
            slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
            r_values.append(r_value)
            p_values.append(p_value)

        if index != 0:
            for data in fig['data']:
                data['showlegend'] = False
        data_groups.append(fig['data'])
        
    # Reference: https://www.dummies.com/education/math/statistics/what-a-p-value-tells-you-about-statistical-data/
    # p-value lesser than 0.05 indicates strong evidence against null hypothesis
    subplot_titles = [f'r2: {r2 :.6f} | p: {p_values[i] :.3f}' for i,r2 in enumerate(np.square(r_values))] if trendline else None
    
    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        xaxis_titles=[xy[0] for xy in xy_tuples],
                        yaxis_titles=[xy[1] for xy in xy_tuples],
                        subplot_titles=subplot_titles,
                        layout_kwargs=layout_kwargs, to_image=to_image)

def pair(df, title='Pair', color=None,
         out_path=None, layout_kwargs={}, to_image=False,
         traces_kwargs={}):
    
    columns = df.select_dtypes(include='number')
    columns = [x for x in columns if x != color]
    fig     = px.scatter_matrix(df, dimensions=columns, color=color)

    layout_kwargs['title'] = title
    fig.update_layout(**layout_kwargs)
    fig.update_traces(**traces_kwargs)

    generate_plot(fig, out_path=out_path, out_filename=title, to_image=to_image)

def heatmap(x, y, z, title='Heatmap',
            out_path=None, layout_kwargs={}, to_image=False,
            heatmap_kwargs={}):

    data = go.Heatmap(
        x=x,
        y=y,
        z=z,
        **heatmap_kwargs
    )
    fig = go.Figure(data=data)

    layout_kwargs['title'] = title
    fig.update_layout(**layout_kwargs)

    generate_plot(fig, out_path=out_path, out_filename=title, to_image=to_image)

def corrmat(df, title='Correlation Matrix',
            out_path=None, layout_kwargs={}, to_image=False,
            heatmap_kwargs={}, matrix_type='default', absolute=False):

    matrix_types = ['default', 'upper', 'lower']
    assert matrix_type in matrix_types, f'matrix_type not in valid list: {matrix_types}'

    corr_df  = df.corr().abs() if absolute else df.corr()
    upper_df = corr_df.where(np.triu(np.ones(corr_df.shape), k=0).astype(np.bool))
    lower_df = corr_df.where(np.tril(np.ones(corr_df.shape), k=0).astype(np.bool))

    heatmap_df = upper_df if matrix_type == 'upper' else lower_df if matrix_type == 'lower' else corr_df
    heatmap_kwargs['transpose'] = True

    heatmap(
        x=heatmap_df.columns,
        y=heatmap_df.index,
        z=heatmap_df.values,
        layout_kwargs=layout_kwargs,
        heatmap_kwargs=heatmap_kwargs,
        title=title,
        out_path=out_path,
        to_image=to_image
    )

def distmat(df, target, title='Distribution Matrix',
            out_path=None, layout_kwargs={}, to_image=False,
            heatmap_kwargs={}):
    
    columns = df.select_dtypes(include='object')
    columns = [x for x in columns if x != target]

    dist_dfs = []
    for column in columns:
        dist_df = pd.crosstab(index=df[column], columns=df[target])
        dist_df['DISTMAT_TOTAL'] = dist_df.sum(axis=1)
        dist_df = pd.concat([(dist_df[x] / dist_df['DISTMAT_TOTAL']).to_frame(name=x) for x in dist_df.columns if x != 'DISTMAT_TOTAL'], axis=1)
        dist_df.index = f'{column}_' + dist_df.index
        dist_dfs.append(dist_df)

    dist_df = pd.concat(dist_dfs)
    heatmap(
        x=dist_df.columns,
        y=dist_df.index,
        z=dist_df.values,
        layout_kwargs=layout_kwargs,
        heatmap_kwargs=heatmap_kwargs,
        title=title,
        out_path=out_path,
        to_image=to_image
    )

# Reference: https://stackoverflow.com/questions/38452379/plotting-a-dendrogram-using-plotly-python
def dendrogram(df, title='Dendrogram',
               out_path=None, layout_kwargs={}, to_image=False):

    fig = ff.create_dendrogram(df, linkagefun=lambda x: linkage(df, method='ward', metric='euclidean'))

    layout_kwargs['title'] = title
    fig.update_layout(**layout_kwargs)

    generate_plot(fig, out_path=out_path, out_filename=title, to_image=to_image)



# AUDIO PROCESSING
def wave(df, amplitude, title='Wave',
         out_path=None, max_col=2, layout_kwargs={}, to_image=False):
    
    new_df = df.copy()
    new_df.rename(columns={amplitude: 'WAVE_AMPLITUDE'}, inplace=True)

    data_groups = []
    for row in new_df.itertuples():
        tmp_df = pd.DataFrame({
            'WAVE_AMPLITUDE': row.WAVE_AMPLITUDE
        })
        
        fig = px.line(tmp_df, y='WAVE_AMPLITUDE')
        data_groups.append(fig['data'])

    xaxis_titles = 'Time (' + (new_df['rate'].astype(int).astype(str) + ' Hz/s').values + ')' \
                   if 'rate' in new_df.columns else ['Time' for _ in new_df.index]
    yaxis_titles = ['Amplitude' if i % max_col == 0 else '' for i,_ in enumerate(new_df.index)]

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        subplot_titles=new_df.index,
                        xaxis_titles=xaxis_titles,
                        yaxis_titles=yaxis_titles,
                        layout_kwargs=layout_kwargs, to_image=to_image)

def fourier(df, frequency, magnitude, title='FT', x_title='Frequency',
            out_path=None, max_col=2, layout_kwargs={}, to_image=False):
    
    new_df = df.copy()
    new_df.rename(columns={
        frequency: 'FT_FREQUENCY',
        magnitude: 'FT_MAGNITUDE'
    }, inplace=True)

    data_groups = []
    for row in new_df.itertuples():
        tmp_df = pd.DataFrame({
            'FT_FREQUENCY': row.FT_FREQUENCY,
            'FT_MAGNITUDE': row.FT_MAGNITUDE
        })
        
        fig = px.line(tmp_df, x='FT_FREQUENCY', y='FT_MAGNITUDE')
        data_groups.append(fig['data'])

    xaxis_titles = [x_title for _ in new_df.index]
    yaxis_titles = ['Magnitude' if i % max_col == 0 else '' for i,_ in enumerate(new_df.index)]

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        subplot_titles=new_df.index,
                        xaxis_titles=xaxis_titles,
                        yaxis_titles=yaxis_titles,
                        layout_kwargs=layout_kwargs, to_image=to_image)

def spectogram(df, z, title='Spectogram', y_title='Frequency',
               out_path=None, max_col=2, layout_kwargs={}, to_image=False):
    
    new_df = df.copy()
    new_df.rename(columns={z: 'SPECTOGRAM_Z'}, inplace=True)

    data_groups = []
    for row in new_df.itertuples():
        data = go.Heatmap(
            z=row.SPECTOGRAM_Z,
            showscale=False
        )
        fig = go.Figure(data=data)

        data_groups.append(fig['data'])

    xaxis_titles = ['Time' for _ in new_df.index]
    yaxis_titles = [y_title if i % max_col == 0 else '' for i,_ in enumerate(new_df.index)]

    datagroups_subplots(data_groups, max_col=max_col, title=title, out_path=out_path,
                        subplot_titles=new_df.index,
                        xaxis_titles=xaxis_titles,
                        yaxis_titles=yaxis_titles,
                        layout_kwargs=layout_kwargs, to_image=to_image)