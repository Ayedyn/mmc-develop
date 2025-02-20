
cd('/drives/mobi1/users/aidenlewis/optix_immc_projects/optix-shijiever/mmc/examples/optix-IMMC_Matlab_Inputs_test')
addpath(genpath('/drives/mobi1/users/aidenlewis/optix_immc_projects/optix-shijiever'));

%% Prep a 60 by 60 by 60 mm cube with a 10mm sphere embedded
% create a surface mesh for a 10 mm radius sphere

[nbox,ebox]=meshgrid6(0:60:60,0:60:60,0:60:60);
fbox=volface(ebox);
ebox(:,5)=1;

%% create simulation parameters
%%-----------------------------------------------------------------
clear cfg

cfg.nphoton=1e8;
cfg.seed=1648335518;
cfg.srcpos=[30,30,0.001];
cfg.srcdir=[0 0 1];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-9;
cfg.prop=[0 0 1 1;
    0.005,  1, 0, 1.37;
    0.001,  1, 0, 1.37
    ];
cfg.debuglevel='TP';
cfg.isreflect=0;
cfg.node = nbox;
cfg.elem = ebox;
%placeholder for tiny sphere at corner of sim
cfg.spheres = [63, 63, 63, 0.1];
cfg.capsulecenters = [30, 30, 30; 40, 40, 30];
cfg.capsulewidths = [5];
cfg.outputtype = 'fluence';
cfg.compute = 'optix';

mmclab(cfg);