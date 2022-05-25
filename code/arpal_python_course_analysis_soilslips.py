#!/usr/bin/python3
"""
ARPAL Labs - SOIL SLIPS ANALYSIS

__date__ = '20210702'
__version__ = '1.0.0'
__author__ =
        'Fabio Delogu (fabio.delogu@cimafoundation.org',

__library__ = 'ARPAL'

General command line:
python3 arpal_soilslips_analysis.py -settings_file configuration.json

Version(s):
20210702 (1.0.0) --> Beta release
"""

# -------------------------------------------------------------------------------------
# Complete library
import os
import json
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Algorithm information
alg_version = '1.0.0'
alg_release = '2021-07-02'
alg_name = 'SOIL SLIPS ANALYSIS'
# Algorithm parameter(s)
time_format_datasets = '%Y-%m-%d'
time_format_graph = '%y-%m-%d'
# Algorithm file settings
file_name_settings = 'arpal_python_course_analysis_soilslips.json'
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# Script Main
def main():
    # -------------------------------------------------------------------------------------
    # NOTEBOOK SETTINGS
    # Open file settings
    with open(file_name_settings) as file_handle_settings:
        file_data_settings = json.load(file_handle_settings)

    # Get settings datasets
    flag_plot_sm_rain = file_data_settings['flags']['graph_sa_and_rain']

    alert_area_name = file_data_settings['parameters']['alert_area_name']

    time_start = file_data_settings['time_period']['time_start']
    time_end = file_data_settings['time_period']['time_end']

    folder_name_scenarios = file_data_settings['data']['scenarios']['folder_name']
    file_name_scenarios = file_data_settings['data']['scenarios']['file_name']

    seasons_lut = {
            1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
            7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # NOTEBOOK DATASETS
    dframe_scenarios_csv = pd.read_csv(os.path.join(folder_name_scenarios, file_name_scenarios))
    dframe_scenarios_csv = dframe_scenarios_csv.set_index('time')
    dframe_scenarios_csv = dframe_scenarios_csv.iloc[::-1]

    # Add season reference
    grp_season = [seasons_lut.get(pd.Timestamp(t_stamp).month) for t_stamp in dframe_scenarios_csv.index.values]
    dframe_scenarios_csv['event_season'] = grp_season
    # Print dataframe head
    print(dframe_scenarios_csv.head(5))
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # EXERCISE 5
    dframe_scenarios_5 = dframe_scenarios_csv.copy()

    event_index_min = 2

    time_start = None # '2018-01-01'
    time_end = None # '2019-01-01'

    alert_area_name = None # 'alert_area_c'

    line_data_x = np.linspace(0, 1, 101)

    line_data_y_thr1 = -200 * line_data_x + 100
    line_data_y_thr1[line_data_y_thr1 < 0] = 0.0

    line_data_y_thr2 = -90 * line_data_x + 100
    line_data_y_thr2[line_data_y_thr2 < 0] = 10.0

    # Select alert area datasets
    if alert_area_name is not None:
        dframe_scenarios_5 = dframe_scenarios_5.loc[dframe_scenarios_5['event_domain'] == alert_area_name]

    # Select time period datasets
    if (time_start is not None) and (time_end is not None):
        time_stamp_start = pd.Timestamp(time_start)
        time_stamp_end = pd.Timestamp(time_end)
        dframe_scenarios_5 = dframe_scenarios_5[time_stamp_start.strftime(time_format_datasets):
                                                time_stamp_end.strftime(time_format_datasets)]

    if event_index_min:
        dframe_scenarios_5 = dframe_scenarios_5[dframe_scenarios_5["event_index"] >= event_index_min]

    var_data_x = dframe_scenarios_5['sm_value_first'].values
    var_data_y = dframe_scenarios_5['rain_accumulated_6H'].values

    var_data_thr1 = []
    var_data_thr2 = []
    for var_point_x, var_point_y in zip(var_data_x, var_data_y):

        line_index_x = np.abs(line_data_x - var_point_x).argmin()

        line_point_x = line_data_x.flat[line_index_x]
        line_point_y_thr1 = line_data_y_thr1.flat[line_index_x]
        line_point_y_thr2 = line_data_y_thr2.flat[line_index_x]

        var_point_thr1 = 0
        if (np.isfinite(line_point_y_thr1)) and (var_point_y > line_point_y_thr1):
            var_point_thr1 = 1
        if (np.isfinite(line_point_y_thr1)) and (var_point_x == 1):
            if var_point_y < 10:
                var_point_thr1 = 1
            else:
                var_point_thr1 = 0

        var_point_thr2 = 0
        if (np.isfinite(line_point_y_thr2)) and (var_point_y > line_point_y_thr2):
            if var_point_y >= 10:
                var_point_thr2 = 1

        # To assign only a class
        if (var_point_thr1 == 1) and (var_point_thr2 == 1):
            var_point_thr1 = 0

        var_data_thr1.append(var_point_thr1)
        var_data_thr2.append(var_point_thr2)

    dframe_scenarios_5['event_thr1'] = var_data_thr1
    dframe_scenarios_5['event_thr2'] = var_data_thr2

    dframe_scenarios_5_thr1 = dframe_scenarios_5[dframe_scenarios_5["event_thr1"] == 1]
    dframe_scenarios_5_thr2 = dframe_scenarios_5[dframe_scenarios_5["event_thr2"] == 1]
    dframe_scenarios_5_thr0 = dframe_scenarios_5[(dframe_scenarios_5["event_thr1"] == 0) & (dframe_scenarios_5["event_thr2"] == 0)]

    grade_thr0 = dframe_scenarios_5_thr0['event_index'].value_counts(normalize=True) * 100
    grade_thr1 = dframe_scenarios_5_thr1['event_index'].value_counts(normalize=True) * 100
    grade_thr2 = dframe_scenarios_5_thr2['event_index'].value_counts(normalize=True) * 100

    grade_default = {0: None, 1: None, 2: None, 3: None, 4: None}

    grade_thr0 = {**grade_default, **grade_thr0}
    grade_thr1 = {**grade_default, **grade_thr1}
    grade_thr2 = {**grade_default, **grade_thr2}

    grade_dict = {'thr0': grade_thr0, 'thr1':grade_thr1, 'thr2': grade_thr2}

    dframe_grade = pd.DataFrame(grade_dict)
    print(dframe_grade)

    # Open figure
    fig = dframe_grade.transpose().plot.bar()

    fig.set_xlabel('event group', color='#000000', fontsize=14, fontdict=dict(weight='medium'))
    fig.set_ylabel('event percentage [%]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))

    fig.grid(b=False, color='grey', linestyle='-', linewidth=0.5, alpha=1)

    fig.set_title(' #### Event distribution according with thresholds #### ',
                   fontdict=dict(fontsize=10, fontweight='bold'))


    print('ciao')

    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # EXERCISE 3
    dframe_scenarios_3 = dframe_scenarios_csv.copy()

    event_label = True
    event_label_min = 2
    event_index_min = 2

    var_x_limits = [0, 1]
    var_y_limits = [0, 100]
    var_z_limits = [0, 4]

    season_label = 'ALL'

    time_start = None # '2018-01-01'
    time_end = None # '2019-01-01'

    alert_area_name = None # 'alert_area_c'

    # Select alert area datasets
    if alert_area_name is not None:
        dframe_scenarios_3 = dframe_scenarios_3.loc[dframe_scenarios_3['event_domain'] == alert_area_name]

    # Select time period datasets
    if (time_start is not None) and (time_end is not None):
        time_stamp_start = pd.Timestamp(time_start)
        time_stamp_end = pd.Timestamp(time_end)
        dframe_scenarios_3 = dframe_scenarios_3[time_stamp_start.strftime(time_format_datasets):
                                                time_stamp_end.strftime(time_format_datasets)]

    dframe_scenarios_3.loc[dframe_scenarios_3['event_index'] < event_index_min] = np.nan
    dframe_scenarios_3 = dframe_scenarios_3.dropna(how='all')
    print(dframe_scenarios_3.head(30))

    var_time = dframe_scenarios_3.index.values.tolist()
    var_data_x = dframe_scenarios_3['sm_value_first'].values
    var_data_y = dframe_scenarios_3['rain_accumulated_6H'].values
    var_data_z = dframe_scenarios_3['event_index'].values

    var_p95_x = np.percentile(var_data_x, 95)
    var_p99_x = np.percentile(var_data_x, 99)

    var_p95_str = '{0:.2f}'.format(var_p95_x)
    var_p99_str = '{0:.2f}'.format(var_p99_x)

    var_time_from_str = pd.Timestamp(var_time[0]).strftime('%Y-%m-%d')
    var_time_to_str = pd.Timestamp(var_time[-1]).strftime('%Y-%m-%d')

    line_point_x = np.linspace(0, 1, 100)

    # Open figure
    fig = plt.figure(figsize=(17, 11))
    fig.autofmt_xdate()

    axes = plt.axes()
    axes.autoscale(True)

    p95 = axes.axvline(var_p95_x, color='#FFA500', linestyle='-', lw=2, label='95%')
    plt.text(var_p95_x, -0.02, var_p95_str, transform=axes.get_xaxis_transform(), ha='center', va='center')
    p99 = axes.axvline(var_p99_x, color='#FF0000', linestyle='-', lw=2, label='99%')
    plt.text(var_p99_x, -0.02, var_p99_str, transform=axes.get_xaxis_transform(), ha='center', va='center')

    colors = {0: 'grey', 1: 'green', 2: 'yellow', 3: 'orange', 4: 'red'}
    for t, x, y, z in zip(var_time, var_data_x, var_data_y, var_data_z):

        if z >= event_index_min:
            t = pd.Timestamp(t)

            if y >= 0:
                color = colors[z]
                p1 = axes.scatter(x, y, alpha=1, color=color, s=20)

                if event_label:
                    if z > event_label_min:
                        label = t.strftime('%Y-%m-%d')
                        plt.annotate(label,  # this is the text
                                     (x, y),  # this is the point to label
                                     textcoords="offset points",  # how to position the text
                                     xytext=(0, 5),  # distance from text to points (x,y)
                                     ha='center')  # horizontal alignment can be left, right or center
            else:
                print(' ===> Value of y is negative (' + str(y) + ') at time ' + str(t))

    line_point_y = -90 * line_point_x + 100
    line_point_y[line_point_y < 0] = np.nan
    line_index_nan = np.argwhere(np.isnan(line_point_y))
    line_index_nan = line_index_nan.flatten().tolist()
    plt.plot(line_point_x, line_point_y, '-r', label='')

    line_point_y = -125 * line_point_x + 75
    line_point_y[line_point_y < 0] = np.nan
    #plt.plot(line_point_x, line_point_y, '-.g', label='')

    line_point_y = -150 * line_point_x + 150
    line_point_y[line_point_y < 0] = np.nan
    #plt.plot(line_point_x, line_point_y, ':b', label='')

    line_point_y = -200 * line_point_x + 100
    line_point_y[line_point_y < 0] = np.nan
    plt.plot(line_point_x, line_point_y, '--m', label='')

    axes.set_xlabel('soil moisture first [-]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))
    axes.set_xlim(var_x_limits[0], var_x_limits[1])
    axes.set_ylabel('rain accumulated 6H [mm]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))
    #if var_y_limits is not None:
    #    axes.set_ylim(var_y_limits[0], var_y_limits[1])


    xticks_list = axes.get_xticks().tolist()
    xticks_list.insert(0, -0.01)
    xticks_list.insert(len(xticks_list), 1.01)
    axes.set_xticks(xticks_list)
    axes.set_xticklabels(['', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0', ''], fontsize=12)

    axes.set_ylim(0, 100)
    axes.set_yscale('log')

    legend = axes.legend((p95, p99), ('95%', '99%'), frameon=True, ncol=3, loc=9)
    axes.add_artist(legend)

    axes.grid(b=False, color='grey', linestyle='-', linewidth=0.5, alpha=1)

    axes.set_title(' #### Scenarios - Rain and Soil Moisture #### \n ' +
                   'TimePeriod :: ' + var_time_from_str + ' - ' + var_time_to_str + ' Season:: ' + season_label,
                   fontdict=dict(fontsize=16, fontweight='bold'))
    plt.show()
    # -------------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------------
    # EXERCISE 4
    dframe_scenarios_4 = dframe_scenarios_csv.copy()
    event_index_min = 3

    dframe_scenarios_4.loc[dframe_scenarios_4['event_index'] < event_index_min] = np.nan
    dframe_scenarios_4 = dframe_scenarios_4.dropna(how='all')
    print(dframe_scenarios_4.head(30))

    var_data_x = dframe_scenarios_4['sm_value_first'].values
    var_data_y = dframe_scenarios_4['rain_accumulated_6H'].values

    fig = plt.figure(figsize=(17, 11))
    axes = plt.axes()
    axes.autoscale(True)

    n, bins, patches = plt.hist(var_data_x, 8)

    axes.set_xlim(0, 1)
    plt.grid(True)

    axes.set_xlabel('soil moisture first [-]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))
    axes.set_ylabel('frequency [-]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))

    plt.show()

    fig = plt.figure(figsize=(17, 11))
    axes = plt.axes()
    axes.autoscale(True)

    plt.hist2d(var_data_x, var_data_y, bins=8, cmap='Blues')
    cb = plt.colorbar()
    cb.set_label('counts in bin')

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 100)

    axes.set_xlabel('soil moisture first [-]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))
    axes.set_ylabel('rain accumulated 6H [mm]', color='#000000', fontsize=14, fontdict=dict(weight='medium'))

    plt.show()
    # -------------------------------------------------------------------------------------


    # -------------------------------------------------------------------------------------
    # EXERCISE 2
    dframe_scenarios_2 = dframe_scenarios_csv.copy()

    # Select alert area datasets
    if alert_area_name is not None:
        dframe_scenarios_2 = dframe_scenarios_2.loc[dframe_scenarios_2['event_domain'] == alert_area_name]

    # Select time period datasets
    if (time_start is not None) and (time_end is not None):
        time_stamp_start = pd.Timestamp(time_start)
        time_stamp_end = pd.Timestamp(time_end)
        dframe_scenarios_2 = dframe_scenarios_2[time_stamp_start.strftime(time_format_datasets):
                                            time_stamp_end.strftime(time_format_datasets)]

    # Get time
    tick_time_period = list(dframe_scenarios_2.index.values)
    tick_time_idx = range(0, tick_time_period.__len__())
    tick_time_labels = [pd.Timestamp(tick_label).strftime(time_format_datasets) for tick_label in tick_time_period]

    dframe_scenarios_2['rain_reset_step'] = 1
    dframe_scenarios_2['rain_reset_step'] = dframe_scenarios_2['rain_reset_step'].where(dframe_scenarios_2['rain_accumulated_6H'] < 2, 0)
    dframe_scenarios_2['rain_reset_cumsum'] = dframe_scenarios_2['rain_reset_step'].cumsum()

    dframe_scenarios_2['rain_cumsum_6H'] = dframe_scenarios_2.groupby(['rain_reset_cumsum'])['rain_accumulated_6H'].cumsum()
    dframe_scenarios_2['event_index_cumsum'] = dframe_scenarios_2.groupby(['rain_reset_cumsum'])['event_index'].cumsum()

    print(dframe_scenarios_2.head(30))

    series_rain_peak_3H = dframe_scenarios_2['rain_peak_3H']
    series_rain_accum_6H = dframe_scenarios_2['rain_accumulated_6H']
    series_event_index = dframe_scenarios_2['event_index']
    series_rain_cumsum_6H = dframe_scenarios_2['rain_cumsum_6H']
    series_cumsum_index = dframe_scenarios_2['event_index_cumsum']

    fig = plt.figure(figsize=(17, 11))
    fig.autofmt_xdate()

    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xticks(tick_time_idx)
    ax1.set_xticklabels(tick_time_labels, rotation=90, fontsize=8)
    # ax1.set_xlim(tick_time_start, tick_time_end)
    ax1.grid(b=True)

    p1 = ax1.plot(series_event_index.index, series_event_index.values,
                  'o', color='red', lw=2)

    ax1.set_ylabel('event index [-]', color='#000000')
    ax1.set_ylim(0, 15)

    ax2 = ax1.twinx()

    ax2.set_ylabel('rain', color='#000000')
    ax2.set_ylim(0, 200)

    ax2.set_xticks(tick_time_idx)

    x_n = 100
    x_min, x_max = ax2.get_xlim()
    x_tmp = np.linspace(x_min, x_max, x_n, dtype=int)

    x_labels = []
    x_ticks = []
    for x_step in x_tmp:
        if (x_step >= 0) and (x_step <= tick_time_labels.__len__()):
            x_label = tick_time_labels[x_step]
            x_ticks.append(x_step)
            x_labels.append(x_label)

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)

    # ax2.set_xticklabels([])
    # ax2.set_xlim(tick_time_start, tick_time_end)

    p2 = ax2.bar(series_rain_cumsum_6H.index, series_rain_cumsum_6H.values,
                  color='blue', lw=2, linestyle='--')
    # p3 = ax2.plot(series_rain_peak_3H.index, series_rain_peak_3H.values, 'o', color='red', lw=2)

    legend = ax1.legend((p1[0], p2[0]),
                        ('event index', 'rain accumulated', 'rain peak',),
                        frameon=False, loc=2)
    ax1.add_artist(legend)
    ax1.set_title('Soil moisture and Rain plot')

    plt.show()

    print('ciao')

    # -------------------------------------------------------------------------------------
    # EXERCISE 1
    dframe_scenarios_1 = dframe_scenarios_csv.copy()

    # Select alert area datasets
    if alert_area_name is not None:
        dframe_scenarios_1 = dframe_scenarios_1.loc[dframe_scenarios_1['event_domain'] == alert_area_name]

    # Select time period datasets
    if (time_start is not None) and (time_end is not None):
        time_stamp_start = pd.Timestamp(time_start)
        time_stamp_end = pd.Timestamp(time_end)
        dframe_scenarios_1 = dframe_scenarios_1[time_stamp_start.strftime(time_format_datasets):
                                            time_stamp_end.strftime(time_format_datasets)]

    # Get series datasets
    series_sm_first = dframe_scenarios_1['sm_value_first']
    series_sm_last = dframe_scenarios_1['sm_value_last']
    series_rain_acc_3H = dframe_scenarios_1['rain_accumulated_3H']
    series_rain_acc_6H = dframe_scenarios_1['rain_accumulated_6H']

    # Get time
    tick_time_period = list(dframe_scenarios_1.index.values)
    tick_time_idx = range(0, tick_time_period.__len__())
    tick_time_labels = [pd.Timestamp(tick_label).strftime(time_format_datasets) for tick_label in tick_time_period]

    tick_time_start = pd.Timestamp(tick_time_period[0]).strftime(time_format_datasets)
    tick_time_end = pd.Timestamp(tick_time_period[-1]).strftime(time_format_datasets)

    # Plot datasets
    if flag_plot_sm_rain:
        fig = plt.figure(figsize=(17, 11))
        fig.autofmt_xdate()

        ax1 = plt.subplot(1, 1, 1)
        ax1.set_xticks(tick_time_idx)
        ax1.set_xticklabels(tick_time_labels, rotation=90, fontsize=8)
        #ax1.set_xlim(tick_time_start, tick_time_end)
        ax1.grid(b=True)

        p1 = ax1.plot(series_sm_first.index, series_sm_first.values, color='#DA70D6', linestyle='--', lw=2)

        ax1.set_ylabel('soil moisture [-]', color='#000000')
        ax1.set_ylim(0, 1)

        ax2 = ax1.twinx()

        ax2.set_ylabel('rain', color='#000000')
        ax2.set_ylim(0, 200)

        ax2.set_xticks(tick_time_idx)

        x_n = 20
        x_min, x_max = ax2.get_xlim()
        x_tmp = np.linspace(x_min, x_max, x_n, dtype=int)

        x_labels = []
        x_ticks = []
        for x_step in x_tmp:
            if (x_step >= 0) and (x_step <= tick_time_labels.__len__()):
                x_label = tick_time_labels[x_step]
                x_ticks.append(x_step)
                x_labels.append(x_label)

        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels)

        # ax2.set_xticklabels([])
        # ax2.set_xlim(tick_time_start, tick_time_end)

        p2 = ax2.bar(series_rain_acc_3H.index, series_rain_acc_3H.values, color='blue', lw=2)

        legend = ax1.legend((p1[0], p2[0]),
                            ('soil_moisture', 'rain',),
                            frameon=False, loc=2)
        ax1.add_artist(legend)
        ax1.set_title('Soil moisture and Rain plot')

        plt.show()
    # -------------------------------------------------------------------------------------



    print('ciao')

# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Call script from external library
if __name__ == '__main__':
    main()
# -------------------------------------------------------------------------------------
