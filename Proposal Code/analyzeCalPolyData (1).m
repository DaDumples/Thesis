% Script designed to help Cal Poly SLO extract LM SSA data from a formatted
% CSV file

% Workspace management
close all; clear all; clc;

%% File Parsing

% New entries will be appended to this file over time
fileName = 'CalPolySLO_obs.csv';
% Data format:
% Constants (beginning of each line):
%  - detection request ID (unique LM identifier)
%  - satellite NORAD ID
%  - epoch of first measurement "yyyy-MM-dd HH:mm:ss.SSS"
% Measurements (repeating sequence)
%  - delta-t from measurement epoch [msec]
%  - observed az [deg]
%  - observed el (corrected for refraction) [deg]
%  - observed RA J2000.0 [deg]
%  - observed Dec J2000.0 [deg]
%  - [OPTIONAL] photometric magnitude

% Skip over the first 13 lines of the file, which contain a header
% describing the data format
fid = fopen(fileName, 'r');
for i = 1:14
    tline = fgetl(fid);
end

% Read each line from the CSV; parse out the constants, then save repeating
% measurement sequence in a data structure
req = struct();
while ischar(tline)
    % The first 3 elements contain the detection request ID, NORAD ID, and
    % epoch of first measurement
    k = strfind(tline, ',');
    C = textscan(tline(1:k(3)), '%f%f%s', 'Delimiter',',');
    requestID = C{1};
    noradID = C{2};
    dnEpoch = datenum(C{3},'yyyy-mm-dd HH:MM:SS.FFF');
    % The remaining entries repeat a 6-element data sequence consisting of
    % the time offset from measurement epoch (msec), measurement azimuth
    % (deg), measurement geometric elevation (deg), measurement right
    % ascension at epoch J2000.0 (deg), measurement declination at epoch
    % J2000.0 (deg), and [OPTIONAL] measurement photometric magnitude
    C = textscan(tline(k(3)+1:end), '%f%f%f%f%f%f', ...
        'Delimiter',',', 'MultipleDelimsAsOne',false, 'EmptyValue',NaN);
    % Save data indexed by request ID (unique identifier)
    elementName = sprintf('r%d', requestID);
    req.(elementName) = struct('requestID',requestID, 'noradID',noradID, ...
        'dnEpoch',dnEpoch, 'dtMsec',C{1}, 'azDeg',C{2}, 'elDeg',C{3}, ...
        'raDeg',C{4}, 'decDeg',C{5}, 'photoMag',C{6}, 'dn',[]);
    % Next line
    tline = fgetl(fid);
end
fclose(fid);

% Cleanup workspace
clear i fid tline k C requestID noradID dnEpoch elementName ans;

%% Post-Processing

% Convert the time offset of each measurement into an incremental array of
% Matlab datenumbers
f = fieldnames(req);
for i = 1:length(f)
    n = length(req.(f{i}).dtMsec);
    req.(f{i}).dn = req.(f{i}).dnEpoch*ones(n, 1);
    % Convert offset in msec to fraction of day
    dt = req.(f{i}).dtMsec./1000./86400;
    req.(f{i}).dn = req.(f{i}).dn + dt;
end
% Cleanup workspace
clear dt;

%% Plotting

for i = 1:length(f)
    data = req.(f{i});
    n = length(data.dtMsec);
    titleCell = {sprintf('NORAD %d',data.noradID), ...
        sprintf('%d obs starting %s UTC',n,datestr(data.dnEpoch,'yyyy-mm-dd HH:MM:SS'))};
    
    % Light curve
    figure;
    plot(data.dtMsec/1000, data.photoMag, 'b.');
    grid on;
    xlabel('measurement offset (sec)');
    ylabel('photometric magnitude');
    title(titleCell);
    % Can be more intuitive to view axis direction by increasing brightness
    set(gca(), 'YDir','reverse');
    
    % Just showing an example for a single satellite pass
    break;
end
% Cleanup workspace
clear i data n titleCell;
