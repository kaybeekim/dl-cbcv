print('plot.py is loaded.')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np

def plot_time_series(actual_df, predicted_df, time_col, value_col, title, 
                     formatter=None, figsize=(20, 6), xlabel='Week', ylabel=None, xticks_interval=None):
    actual_df['time'] = pd.to_datetime(actual_df['time'], format='%Y-%m-%d')
    predicted_df['time'] = pd.to_datetime(predicted_df['time'], format='%Y-%m-%d')
    
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)
    if xticks_interval:
        plt.xticks(np.arange(0, len(actual_df[time_col]), xticks_interval), rotation=45)
        
    unit=1
    if formatter:
        if actual_df[value_col].mean() > 1000000:
            def millions(x, pos):
                return '%1.1f M' % (x)
            formatter = ticker.FuncFormatter(millions)
            unit=1e6
            plt.gca().yaxis.set_major_formatter(formatter)
        elif actual_df[value_col].mean() > 1000:
            def thousands(x, pos):
                return '%1.1f K' % (x)  # Multiply by 1e-3 to convert from units to thousands
            formatter = ticker.FuncFormatter(thousands)
            unit=1e3
            plt.gca().yaxis.set_major_formatter(formatter)
    plt.plot(actual_df[time_col], actual_df[value_col]/unit, label='Actual')
    plt.plot(predicted_df[time_col], predicted_df[value_col]/unit, label='Predicted')
    plt.legend()



def plot_time_series_multiple(actual_df, predicted_df, time_col, value_col, title, ax, 
                              formatter=None, xlabel='Week', ylabel=None, xticks_interval=None):
    actual_df['time'] = pd.to_datetime(actual_df['time'], format='%Y-%m-%d')
    predicted_df['time'] = pd.to_datetime(predicted_df['time'], format='%Y-%m-%d')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    if xticks_interval:
        ax.set_xticks(np.arange(0, len(actual_df[time_col]), xticks_interval), rotation=45)
        
    unit=1
    if formatter:
        if actual_df[value_col].mean() > 1000000:
            def millions(x, pos):
                return '%1.1f M' % (x)
            formatter = ticker.FuncFormatter(millions)
            unit=1e6
            ax.yaxis.set_major_formatter(formatter)   
        elif actual_df[value_col].mean() > 1000:
            def thousands(x, pos):
                return '%1.1f K' % (x)  # Multiply by 1e-3 to convert from units to thousands
            formatter = ticker.FuncFormatter(thousands)
            unit=1e3
            ax.yaxis.set_major_formatter(formatter)        
    ax.plot(actual_df[time_col], actual_df[value_col]/unit, label='Actual')
    ax.plot(predicted_df[time_col], predicted_df[value_col]/unit, label='Predicted')
    ax.legend()




from matplotlib.backends.backend_pdf import PdfPages

def save_plots_to_pdf(plot_functions, pdf_filename):
    """
    Save multiple plots to a single PDF file.

    :param plot_functions: List of functions, each of which generates a plot.
    :param pdf_filename: Name of the PDF file to save the plots to.
    """
    with PdfPages(pdf_filename) as pdf:
        for plot_func, kwargs in plot_functions:
            # plt.figure()
            plot_func(**kwargs)
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
























