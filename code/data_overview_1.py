# -*- coding: utf-8 -*-
"""
Code for generating the first data figure in the manuscript.

Authors: Julia Sprenger, Lyuba Zehl, Michael Denker


Copyright (c) 2017, Institute of Neuroscience and Medicine (INM-6),
Forschungszentrum Juelich, Germany
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* Neither the names of the copyright holders nor the names of the contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# This loads the Neo and odML libraries shipped with this code. For production
# use, please use the newest releases of odML and Neo.
import load_local_neo_odml_elephant

import os

import numpy as np
from scipy import stats
import quantities as pq
import matplotlib.pyplot as plt

from matplotlib import gridspec, ticker

from reachgraspio import reachgraspio

import odml.tools

import neo_utils
import odml_utils


# =============================================================================
# Define data and metadata directories
# =============================================================================


def get_monkey_datafile(monkey):
    if monkey == "Lilou":
        return "l101210-001"  # ns2 (behavior) and ns5 present
    elif monkey == "Nikos2":
        return "i140703-001"  # ns2 and ns6 present
    else:
        return ""


# Enter your dataset directory here
datasetdir = "../datasets/"

trialtype_colors = {
    'SGHF': 'MediumBlue', 'SGLF': 'Turquoise',
    'PGHF': 'DarkGreen', 'PGLF': 'YellowGreen',
    'LFSG': 'Orange', 'LFPG': 'Yellow',
    'HFSG': 'DarkRed', 'HFPG': 'OrangeRed',
    'SGSG': 'SteelBlue', 'PGPG': 'LimeGreen',
    'NONE': 'k', 'PG': 'k', 'SG': 'k', 'LF': 'k', 'HF': 'k'}

event_colors = {
    'TS-ON': 'Gray',  # 'TS-OFF': 'Gray',
    'WS-ON': 'Gray',  # 'WS-OFF': 'Gray',
    'CUE-ON': 'Gray',
    'CUE-OFF': 'Gray',
    'GO-ON': 'Gray',  # 'GO-OFF': 'Gray',
    #    'GO/RW-OFF': 'Gray',
    'SR': 'Gray',  # 'SR-REP': 'Gray',
    'RW-ON': 'Gray',  # 'RW-OFF': 'Gray',
    'STOP': 'Gray'}


# =============================================================================
# Plot helper functions
# =============================================================================


def force_aspect(ax, aspect=1):
    ax.set_aspect(abs(
        (ax.get_xlim()[1] - ax.get_xlim()[0]) /
        (ax.get_ylim()[1] - ax.get_ylim()[0])) / aspect)


def get_arraygrid(blackrock_elid_list, chosen_el, rej_el=None):
    if rej_el is None:
        rej_el = []
    array_grid = np.zeros((10, 10))
    for m in range(10):
        for n in range(10):
            idx = (9 - m) * 10 + n
            bl_id = blackrock_elid_list[idx]
            if bl_id == -1:
                array_grid[m, n] = 0.7
            elif bl_id == chosen_el:
                array_grid[m, n] = -0.7
            elif bl_id in rej_el:
                array_grid[m, n] = -0.35
            else:
                array_grid[m, n] = 0
    return np.ma.array(array_grid, mask=np.isnan(array_grid))


# =============================================================================
# Load data and metadata for a monkey
# =============================================================================
# CHANGE this parameter to load data of the different monkeys
# monkey = 'Nikos2'
monkey = 'Lilou'

nsx_none = {'Lilou': None, 'Nikos2': None}
nsx_lfp = {'Lilou': 2, 'Nikos2': 2}
nsx_raw = {'Lilou': 5, 'Nikos2': 6}
chosen_el = {'Lilou': 71, 'Nikos2': 63}
chosen_units = {'Lilou': range(1, 5), 'Nikos2': range(1, 5)}

datafile = get_monkey_datafile(monkey)

session = reachgraspio.ReachGraspIO(
    filename=os.path.join(datasetdir, datafile),
    odml_directory=datasetdir,
    verbose=False)

bl_lfp = session.read_block(
    index=None,
    name=None,
    description=None,
    nsx_to_load=nsx_lfp[monkey],
    n_starts=None,
    n_stops=None,
    channels='all',
    units=chosen_units[monkey],
    load_waveforms=False,
    load_events=True,
    scaling='voltage',
    lazy=False,
    cascade=True)

bl_raw = session.read_block(
    index=None,
    name=None,
    description=None,
    nsx_to_load=nsx_raw[monkey],
    n_starts=None,
    n_stops=None,
    channels=chosen_el[monkey],
    units=chosen_units[monkey],
    load_waveforms=True,
    load_events=True,
    scaling='voltage',
    lazy=False,
    cascade=True)

seg_raw = bl_raw.segments[0]
seg_lfp = bl_lfp.segments[0]

# Displaying loaded data structure as string output
print("\nBlock")
print('Attributes ', bl_raw.__dict__.keys())
print('Annotations', bl_raw.annotations)
print("\nSegment")
print('Attributes ', seg_raw.__dict__.keys())
print('Annotations', seg_raw.annotations)
print("\nEvents")
for x in seg_raw.events:
    print('\tEvent with name', x.name)
    print('\t\tAttributes ', x.__dict__.keys())
    print('\t\tAnnotation keys', x.annotations.keys())
    print('\t\ttimes', x.times[:20])
    for anno_key in ['trial_id', 'trial_timestamp_id', 'trial_event_labels',
                     'trial_reject_IFC']:
        print('\t\t'+anno_key, x.annotations[anno_key][:20])

print("\nChannels")
for x in bl_raw.channel_indexes:
    print('\tChannel with name', x.name)
    print('\t\tAttributes ', x.__dict__.keys())
    print('\t\tchannel_ids', x.channel_ids)
    print('\t\tchannel_names', x.channel_names)
    print('\t\tAnnotations', x.annotations)
print("\nUnits")
for x in bl_raw.list_units:
    print('\tUnit with name', x.name)
    print('\t\tAttributes ', x.__dict__.keys())
    print('\t\tAnnotations', x.annotations)
    print('\t\tchannel_id', x.annotations['channel_id'])
    assert(x.annotations['channel_id'] == x.channel_index.channel_ids[0])
print("\nSpikeTrains")
for x in seg_raw.spiketrains:
    print('\tSpiketrain with name', x.name)
    print('\t\tAttributes ', x.__dict__.keys())
    print('\t\tAnnotations', x.annotations)
    print('\t\tchannel_id', x.annotations['channel_id'])
    print('\t\tspike times', x.times[0:20])
print("\nAnalogSignals")
for x in seg_raw.analogsignals:
    print('\tAnalogSignal with name', x.name)
    print('\t\tAttributes ', x.__dict__.keys())
    print('\t\tAnnotations', x.annotations)
    print('\t\tchannel_id', x.annotations['channel_id'])

# get start and stop events of trials
start_events = neo_utils.get_events(
    seg_raw,
    properties={
        'name': 'TrialEvents',
        'trial_event_labels': 'TS-ON',
        'performance_in_trial': 255})
stop_events = neo_utils.get_events(
    seg_raw,
    properties={
        'name': 'TrialEvents',
        'trial_event_labels': 'STOP',
        'performance_in_trial': 255})

# there should only be one event object for these conditions
assert len(start_events) == 1
assert len(stop_events) == 1

# insert epochs between 10ms before TS to 50ms after RW corresponding to trails
neo_utils.add_epoch(
    seg_raw,
    start_events[0],
    stop_events[0],
    pre=-250 * pq.ms,
    post=500 * pq.ms,
    trial_status='complete_trials',
    trial_type=start_events[0].annotations['belongs_to_trialtype'],
    trial_performance=start_events[0].annotations['performance_in_trial'])

# access single epoch of this data_segment
epochs = neo_utils.get_epochs(seg_raw,
                              properties={'trial_status': 'complete_trials'})
assert len(epochs) == 1

# cut segments according to inserted 'complete_trials' epochs and reset trial
#  times
cut_segments_raw = neo_utils.cut_segment_by_epoch(
    seg_raw, epochs[0], reset_time=True)

cut_segments_lfp = neo_utils.cut_segment_by_epoch(
    seg_lfp, epochs[0], reset_time=True)

# =============================================================================
# Define data for overview plots
# =============================================================================
trial_index = {'Lilou': 0, 'Nikos2': 6}

trial_seg_raw = cut_segments_raw[trial_index[monkey]]
trial_seg_lfp = cut_segments_lfp[trial_index[monkey]]

blackrock_elid_list = bl_lfp.annotations['avail_electrode_ids']

# get 'TrialEvents'
event = trial_seg_lfp.events[2]
start = event.annotations['trial_event_labels'].index('TS-ON')
trialx_trty = event.annotations['belongs_to_trialtype'][start]
trialx_trtimeid = event.annotations['trial_timestamp_id'][start]
trialx_color = trialtype_colors[trialx_trty]

# find trial index for next trial with opposite force type (for ax5b plot)
if 'LF' in trialx_trty:
    trialz_trty = trialx_trty.replace('LF', 'HF')
else:
    trialz_trty = trialx_trty.replace('HF', 'LF')

for i, tr in enumerate(cut_segments_lfp):
    eventz = tr.events[2]
    nextft = eventz.annotations['trial_event_labels'].index('TS-ON')
    if eventz.annotations['belongs_to_trialtype'][nextft] == trialz_trty:
        trialz_trtimeid = eventz.annotations['trial_timestamp_id'][nextft]
        trialz_color = trialtype_colors[trialz_trty]
        trialz_seg_lfp = tr
        break


# =============================================================================
# Define figure and subplot axis for first data overview
# =============================================================================
fig = plt.figure()
fig.set_size_inches(6.5, 10.)  # (w, h) in inches

gs = gridspec.GridSpec(
    nrows=5,
    ncols=4,
    left=0.05,
    bottom=0.07,
    right=0.9,
    top=0.975,
    wspace=0.3,
    hspace=0.5,
    width_ratios=None,
    height_ratios=[1, 3, 3, 6, 3])

ax1 = plt.subplot(gs[0, :])  # top row / odml data
# second row
ax2a = plt.subplot(gs[1, 0])  # electrode overview plot
ax2b = plt.subplot(gs[1, 1])  # waveforms unit 1
ax2c = plt.subplot(gs[1, 2])  # waveforms unit 2
ax2d = plt.subplot(gs[1, 3])  # waveforms unit 3
ax3 = plt.subplot(gs[2, :])  # third row / spiketrains
ax4 = plt.subplot(gs[3, :], sharex=ax3)  # fourth row / raw signal
ax5a = plt.subplot(gs[4, 0:3])  # fifth row / behavioral signals
ax5b = plt.subplot(gs[4, 3])

fontdict_titles = {'fontsize': 'small', 'fontweight': 'bold'}
fontdict_axis = {'fontsize': 'x-small'}

wf_time_unit = pq.ms
wf_signal_unit = pq.microvolt

plotting_time_unit = pq.s
raw_signal_unit = wf_signal_unit

behav_signal_unit = pq.V

# =============================================================================
# PLOT TRIAL SEQUENCE OF SUBSESSION
# =============================================================================

# load complete metadata collection
odmldoc = odml.tools.xmlparser.load(datasetdir + datafile + '.odml')

# get total trial number
trno_tot = odml_utils.get_TrialCount(odmldoc)
trno_ctr = odml_utils.get_TrialCount(odmldoc, performance_code=255)
trno_ertr = trno_tot - trno_ctr

# get trial id of chosen trial (and next trial with opposite force)
trtimeids = odml_utils.get_TrialIDs(odmldoc, idtype='TrialTimestampID')
trids = odml_utils.get_TrialIDs(odmldoc)
trialx_trid = trids[trtimeids.index(trialx_trtimeid)]
trialz_trid = trids[trtimeids.index(trialz_trtimeid)]

# get all trial ids for grip error trials
trids_pc191 = odml_utils.get_trialids_pc(odmldoc, 191)

# get all trial ids for correct trials
trids_pc255 = odml_utils.get_trialids_pc(odmldoc, 255)

# get occurring trial types
octrty = odml_utils.get_OccurringTrialTypes(odmldoc, code=False)

# Subplot 1: Trial sequence
boxes, labels = [], []
for tt in octrty:
    # Plot trial ids of current trial type into trial sequence bar plot
    left = odml_utils.get_trialids_trty(odmldoc, tt)
    height = np.ones_like(left)
    width = 1.
    if tt in ['NONE', 'PG', 'SG', 'LF', 'HF']:
        color = 'w'
    else:
        color = trialtype_colors[tt]

    B = ax1.bar(
        x=left, height=height, width=width, color=color, linewidth=0.001, align='edge')

    # Mark trials of current trial type (left) if a grip error occurred
    x = [i for i in list(set(left) & set(trids_pc191))]
    y = np.ones_like(x) * 2.0
    ax1.scatter(x, y, s=5, color='k', marker='*')
    # Mark trials of current trial type (left) if any other error occurred
    x = [i for i in list(
        set(left) - set(trids_pc255) - set(trids_pc191))]
    y = np.ones_like(x) * 2.0
    ax1.scatter(x, y, s=5, color='gray', marker='*')

    # Collect information for trial type legend
    if tt not in ['PG', 'SG', 'LF', 'HF']:
        boxes.append(B[0])
        if tt == 'NONE':
            # use errors for providing total trial number
            labels.append('total: # %i' % trno_tot)
            # add another box and label for error numbers
            boxes.append(B[0])
            labels.append('* errors: # %i' % trno_ertr)
        else:
            # trial type trial numbers
            labels.append(tt + ': # %i' % len(left))

# mark chosen trial
x = [trialx_trid]
y = np.ones_like(x) * 2.0
ax1.scatter(x, y, s=5, marker='D', color='Red', edgecolors='Red')
# mark next trial with opposite force
x = [trialz_trid]
y = np.ones_like(x) * 2.0
ax1.scatter(x, y, s=5, marker='D', color='orange', edgecolors='orange')


# Generate trial type legend; bbox: (left, bottom, width, height)
leg = ax1.legend(
    boxes, labels, bbox_to_anchor=(0., 1., 0.5, 0.1), loc=3, handlelength=1.1,
    ncol=len(labels), borderaxespad=0., handletextpad=0.4,
    prop={'size': 'xx-small'})
leg.draw_frame(False)

# adjust x and y axis
xticks = [i for i in range(1, 101, 10)] + [100]
ax1.set_xticks(xticks)
ax1.set_xticklabels([str(int(t)) for t in xticks], size='xx-small')
ax1.set_xlabel('trial ID', size='x-small')
ax1.set_xlim(1.-width/2., 100.+width/2.)
ax1.yaxis.set_visible(False)
ax1.set_ylim(0, 3)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(direction='out', top='off')
ax1.set_title('sequence of the first 100 trials', fontdict_titles, y=2)
ax1.set_aspect('equal')


# =============================================================================
# PLOT ELECTRODE POSITION of chosen electrode
# =============================================================================
arraygrid = get_arraygrid(blackrock_elid_list, chosen_el[monkey])
cmap = plt.cm.RdGy

ax2a.pcolormesh(
    np.flipud(arraygrid), vmin=-1, vmax=1, lw=1, cmap=cmap, edgecolors='k',
    shading='faceted')

force_aspect(ax2a, aspect=1)
ax2a.tick_params(
    bottom='off', top='off', left='off', right='off',
    labelbottom='off', labeltop='off', labelleft='off', labelright='off')
ax2a.set_title('electrode pos.', fontdict_titles)


# =============================================================================
# PLOT WAVEFORMS of units of the chosen electrode
# =============================================================================
unit_ax_translator = {1: ax2b, 2: ax2c, 3: ax2d}
unit_type = {1: '', 2: '', 3: ''}

wf_lim = []
# plotting waveform for all spiketrains available
for spiketrain in trial_seg_raw.spiketrains:
    unit_id = spiketrain.annotations['unit_id']
    # get unit type
    if spiketrain.annotations['sua']:
        unit_type[unit_id] = 'SUA'
    elif spiketrain.annotations['mua']:
        unit_type[unit_id] = 'MUA'
    else:
        pass
    # get correct ax
    ax = unit_ax_translator[unit_id]
    # get wf sampling time before threshold crossing
    left_sweep = spiketrain.left_sweep

    # plot waveforms in subplots according to unit id
    for st_id, st in enumerate(spiketrain):
        wf = spiketrain.waveforms[st_id]
        wf_lim.append((np.min(wf), np.max(wf)))
        wf_color = str(
            (st / spiketrain.t_stop).rescale('dimensionless').magnitude)
        times = range(len(wf[0])) * spiketrain.units - left_sweep
        ax.plot(
            times.rescale(wf_time_unit), wf[0].rescale(wf_signal_unit),
            color=wf_color)
        ax.set_xlim(
            times.rescale(wf_time_unit)[0], times.rescale(wf_time_unit)[-1])

# adding xlabels and titles
for unit_id, ax in unit_ax_translator.items():
    ax.set_title('unit %i (%s)' % (unit_id, unit_type[unit_id]),
                 fontdict_titles)
    ax.tick_params(direction='in', length=3, labelsize='xx-small',
                   labelleft='off', labelright='off')
    ax.set_xlabel(wf_time_unit.dimensionality.latex, fontdict_axis)
    xticklocator = ticker.MaxNLocator(nbins=5)
    ax.xaxis.set_major_locator(xticklocator)
    ax.set_ylim(np.min(wf_lim), np.max(wf_lim))
    force_aspect(ax, aspect=1)

# adding ylabel
ax2d.tick_params(labelsize='xx-small', labelright='on')
ax2d.set_ylabel(wf_signal_unit.dimensionality.latex, fontdict_axis)
ax2d.yaxis.set_label_position("right")


# =============================================================================
# PLOT SPIKETRAINS of units of chosen electrode
# =============================================================================
plotted_unit_ids = []

# plotting all available spiketrains
for st in trial_seg_raw.spiketrains:
    unit_id = st.annotations['unit_id']
    plotted_unit_ids.append(unit_id)
    ax3.plot(st.times.rescale(plotting_time_unit),
             np.zeros(len(st.times)) + unit_id,
             'k|')

# setting layout of spiktrain plot
ax3.set_ylim(min(plotted_unit_ids) - 0.5, max(plotted_unit_ids) + 0.5)
ax3.set_ylabel(r'unit ID', fontdict_axis)
ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1))
ax3.yaxis.set_label_position("right")
ax3.tick_params(axis='y', direction='in', length=3, labelsize='xx-small',
                labelleft='off', labelright='on')
ax3.invert_yaxis()
ax3.set_title('spiketrains', fontdict_titles)

# =============================================================================
# PLOT "raw" SIGNAL of chosen trial of chosen electrode
# =============================================================================
# get "raw" data from chosen electrode
assert len(trial_seg_raw.analogsignals) == 1
el_raw_sig = trial_seg_raw.analogsignals[0]

# plotting raw signal trace
ax4.plot(el_raw_sig.times.rescale(plotting_time_unit),
         el_raw_sig.squeeze().rescale(raw_signal_unit),
         color='k')

# setting layout of raw signal plot
ax4.set_ylabel(raw_signal_unit.units.dimensionality.latex, fontdict_axis)
ax4.yaxis.set_label_position("right")
ax4.tick_params(axis='y', direction='in', length=3, labelsize='xx-small',
                labelleft='off', labelright='on')
ax4.set_title('"raw" signal', fontdict_titles)

ax4.set_xlim(trial_seg_raw.t_start.rescale(plotting_time_unit),
             trial_seg_raw.t_stop.rescale(plotting_time_unit))
ax4.xaxis.set_major_locator(ticker.MultipleLocator(base=1))


# =============================================================================
# PLOT EVENTS across ax3 and ax4 and add time bar
# =============================================================================
# find trial relevant events
startidx = event.annotations['trial_event_labels'].index('TS-ON')
stopidx = event.annotations['trial_event_labels'][startidx:].index('STOP') + \
    startidx + 1

for ax in [ax3, ax4]:
    xticks = []
    xticklabels = []
    for ev_id, ev in enumerate(event[startidx:stopidx]):
        ev_labels = event.annotations['trial_event_labels'][startidx:stopidx]
        if ev_labels[ev_id] in event_colors.keys():
            ev_color = event_colors[ev_labels[ev_id]]
            ax.axvline(
                ev.rescale(plotting_time_unit), color=ev_color, zorder=0.5)
            xticks.append(ev.rescale(plotting_time_unit))
            if ev_labels[ev_id] == 'CUE-OFF':
                xticklabels.append('-OFF')
            elif ev_labels[ev_id] == 'GO-ON':
                xticklabels.append('GO')
            else:
                xticklabels.append(ev_labels[ev_id])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='x', direction='out', length=3, labelsize='xx-small',
                   labeltop='off', top='off')

timebar_ypos = ax4.get_ylim()[0] + np.diff(ax4.get_ylim())[0] / 10
timebar_labeloffset = np.diff(ax4.get_ylim())[0] * 0.01
timebar_xmin = xticks[-2] + ((xticks[-1] - xticks[-2]) / 2 - 0.25 * pq.s)
timebar_xmax = timebar_xmin + 0.5 * pq.s

ax4.plot([timebar_xmin, timebar_xmax], [timebar_ypos, timebar_ypos], '-',
         linewidth=3, color='k')
ax4.text(timebar_xmin + 0.25 * pq.s, timebar_ypos + timebar_labeloffset,
         '500 ms', ha='center', va='bottom', size='xx-small', color='k')


# =============================================================================
# PLOT BEHAVIORAL SIGNALS of chosen trial
# =============================================================================
# get behavioral signals
ainp_signals = [nsig for nsig in trial_seg_lfp.analogsignals if
                nsig.annotations['channel_id'] > 96]

ainp_trialz = [nsig for nsig in trialz_seg_lfp.analogsignals if
               nsig.annotations['channel_id'] == 141][0]

# find out what signal to use
trialx_sec = odmldoc['Recording']['TaskSettings']['Trial_%03i' % trialx_trid]

# get correct channel id
trialx_chids = [143]
FSRi = trialx_sec['AnalogEvents'].properties['UsedForceSensor'].values[0]
FSRinfosec = odmldoc['Setup']['Apparatus']['TargetObject']['FSRSensor']
if 'SG' in trialx_trty:
    sgchids = FSRinfosec.properties['SGChannelIDs'].values
    trialx_chids.append(min(sgchids) if FSRi == 1 else max(sgchids))
else:
    pgchids = FSRinfosec.properties['PGChannelIDs'].values
    trialx_chids.append(min(pgchids) if FSRi == 1 else max(pgchids))


# define time epoch
startidx = event.annotations['trial_event_labels'].index('SR')
stopidx = event.annotations['trial_event_labels'].index('OBB')
sr = event[startidx].rescale(plotting_time_unit)
stop = event[stopidx].rescale(plotting_time_unit) + 0.050 * pq.s
startidx = event.annotations['trial_event_labels'].index('FSRplat-ON')
stopidx = event.annotations['trial_event_labels'].index('FSRplat-OFF')
fplon = event[startidx].rescale(plotting_time_unit)
fploff = event[stopidx].rescale(plotting_time_unit)

# define time epoch trialz
startidx = eventz.annotations['trial_event_labels'].index('FSRplat-ON')
stopidx = eventz.annotations['trial_event_labels'].index('FSRplat-OFF')
fplon_trz = eventz[startidx].rescale(plotting_time_unit)
fploff_trz = eventz[stopidx].rescale(plotting_time_unit)

# plotting grip force and object displacement
ai_legend = []
ai_legend_txt = []
for ainp in ainp_signals:
    if ainp.annotations['channel_id'] in trialx_chids:
        ainp_times = ainp.times.rescale(plotting_time_unit)
        mask = (ainp_times > sr) & (ainp_times < stop)
        ainp_ampli = stats.zscore(ainp.magnitude[mask])

        if ainp.annotations['channel_id'] != 143:
            color = 'gray'
            ai_legend_txt.append('grip force')
        else:
            color = 'k'
            ai_legend_txt.append('object disp.')
        ai_legend.append(
            ax5a.plot(ainp_times[mask], ainp_ampli, color=color)[0])

    # get force load of this trial for next plot
    elif ainp.annotations['channel_id'] == 141:
        ainp_times = ainp.times.rescale(plotting_time_unit)
        mask = (ainp_times > fplon) & (ainp_times < fploff)
        force_av_01 = np.mean(ainp.rescale(behav_signal_unit).magnitude[mask])

# setting layout of grip force and object displacement plot
ax5a.set_title('grip force and object displacement', fontdict_titles)
ax5a.yaxis.set_label_position("left")
ax5a.tick_params(direction='in', length=3, labelsize='xx-small',
                 labelleft='off', labelright='on')
ax5a.set_ylabel('zscore', fontdict_axis)
ax5a.legend(
    ai_legend, ai_legend_txt,
    bbox_to_anchor=(0.65, .85, 0.25, 0.1), loc=2, handlelength=1.1,
    ncol=len(labels), borderaxespad=0., handletextpad=0.4,
    prop={'size': 'xx-small'})

# plotting load/pull force of LF and HF trial
force_times = ainp_trialz.times.rescale(plotting_time_unit)
mask = (force_times > fplon_trz) & (force_times < fploff_trz)
force_av_02 = np.mean(ainp_trialz.rescale(behav_signal_unit).magnitude[mask])

bar_width = [0.4, 0.4]
color = [trialx_color, trialz_color]
ax5b.bar([0, 0.6], [force_av_01, force_av_02], bar_width, color=color)

ax5b.set_title('load/pull force', fontdict_titles)
ax5b.set_ylabel(behav_signal_unit.units.dimensionality.latex, fontdict_axis)
ax5b.set_xticks([0, 0.6])
ax5b.set_xticklabels([trialx_trty, trialz_trty], fontdict_axis)
ax5b.yaxis.set_label_position("right")
ax5b.tick_params(direction='in', length=3, labelsize='xx-small',
                 labelleft='off', labelright='on')

# =============================================================================
# PLOT EVENTS across ax5a and add time bar
# =============================================================================
# find trial relevant events
startidx = event.annotations['trial_event_labels'].index('SR')
stopidx = event.annotations['trial_event_labels'].index('OBB')

xticks = []
xticklabels = []
for ev_id, ev in enumerate(event[startidx:stopidx]):
    ev_labels = event.annotations['trial_event_labels'][startidx:stopidx + 1]
    if ev_labels[ev_id] in ['RW-ON']:
        ax5a.axvline(ev.rescale(plotting_time_unit), color='k', zorder=0.5)
        xticks.append(ev.rescale(plotting_time_unit))
        xticklabels.append(ev_labels[ev_id])
    elif ev_labels[ev_id] in ['OT', 'OR', 'DO', 'OBB', 'FSRplat-ON',
                              'FSRplat-OFF', 'HEplat-ON']:
        ev_color = 'k'
        xticks.append(ev.rescale(plotting_time_unit))
        xticklabels.append(ev_labels[ev_id])
        ax5a.axvline(
            ev.rescale(plotting_time_unit), color='k', ls='-.', zorder=0.5)
    elif ev_labels[ev_id] == 'HEplat-OFF':
        ev_color = 'k'
        ax5a.axvline(
            ev.rescale(plotting_time_unit), color='k', ls='-.', zorder=0.5)

ax5a.set_xticks(xticks)
ax5a.set_xticklabels(xticklabels, fontdict_axis, rotation=90)
ax5a.tick_params(axis='x', direction='out', length=3, labelsize='xx-small',
                 labeltop='off', top='off')
ax5a.set_ylim([-2.0, 2.0])

timebar_ypos = ax5a.get_ylim()[0] + np.diff(ax5a.get_ylim())[0] / 10
timebar_labeloffset = np.diff(ax5a.get_ylim())[0] * 0.02
timebar_xmax = xticks[xticklabels.index('RW-ON')] - 0.1 * pq.s
timebar_xmin = timebar_xmax - 0.25 * pq.s


ax5a.plot([timebar_xmin, timebar_xmax], [timebar_ypos, timebar_ypos], '-',
          linewidth=3, color='k')
ax5a.text(timebar_xmin + 0.125 * pq.s, timebar_ypos + timebar_labeloffset,
          '250 ms', ha='center', va='bottom', size='xx-small', color='k')

# add time window of ax5a to ax4
ax4.axvspan(ax5a.get_xlim()[0], ax5a.get_xlim()[1], facecolor=[0.9, 0.9, 0.9],
            zorder=-0.1, ec=None)

# =============================================================================
# SAVE FIGURE
# =============================================================================


fname = 'data_overview_1_%s' % monkey
for file_format in ['eps', 'png', 'pdf']:
    fig.savefig(fname + '.%s' % file_format, dpi=400, format=file_format)
