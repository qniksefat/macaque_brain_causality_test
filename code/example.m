% Minimal example for MATLAB users

clear all

% Load data from the specially prepared matlab files
% PLEASE NOTE: These are not the original data files, and are provided for
% convenience only. These files contain the data structures returned by the
% Python loading routines, converted to a matlab format. Thus, these files
% contain a condensed, interpreted subset of the original data.
% Note that due to file size restrictions in the MATLAB file format, raw 
% signals are saved in separate files, channel by channel.
data = load("../datasets_matlab/i140703-001_lfp-spikes.mat");

% Plot the LFP and spikes of electrode 62 from 3s to 5s
% Note: All annotations of the original Python data objects are saved as
% "an_XXX" in the matlab structs
for i=1:length(data.block.segments{1}.analogsignals)
    if data.block.segments{1}.analogsignals{i}.an_channel_id==62
        % LFP
        % Sampled at 1KHz
        time_axis = [0:1/data.block.segments{1}.analogsignals{i}.sampling_rate:(length(data.block.segments{1}.analogsignals{i}.signal)-1)/data.block.segments{1}.analogsignals{i}.sampling_rate];
        plot(time_axis(3000:5000), data.block.segments{1}.analogsignals{i}.signal(3000:5000), 'k-');
        hold on
        
        % Spikes
        for t=1:length(data.block.segments{1}.spiketrains{i}.times)
            % Spikes are saved in the native resolution of 30kHz
            st = data.block.segments{1}.spiketrains{i}.times(t) / 30000;
            if st>=3 && st<=5
                plot(st,0,'rx');
            end
        end
        xlabel("time [1/" + data.block.segments{1}.analogsignals{i}.sampling_rate_units + "]");
        ylabel("LFP [" + data.block.segments{1}.analogsignals{i}.signal_units + "]");
        title(data.block.segments{1}.analogsignals{i}.description);
    end
end

