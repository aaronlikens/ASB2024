
clear; close all; clc;

my_directory = 'E:\Research\Gaitprint\CLEAN DATA';

young = readmatrix('E:\Research\Gaitprint\CLEAN DATA\S027_G01_D02_B02_T01');
% All in tilt forward. time, left contact, right contact, left thigh, right
% thigh, left shank, right shank, left foot, right foot, left upper arm,
% right upper arm, left forearm, right forearm.
young = young(24001:end, [1 154 245 127 218 142 233 158 249 82 173 97 188]);
young = young(1:2:end, :);

% Same for old
old = readmatrix('E:\Research\Gaitprint\CLEAN DATA\S131_G03_D02_B03_T03');
old = old(24001:end, [1 154 245 127 218 142 233 158 249 82 173 97 188]);
old = old(1:2:end, :);

save("asb2024.mat","young","old")
