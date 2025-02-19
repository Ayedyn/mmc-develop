%% Prep a 60 by 60 by 60 mm cube with a 10mm sphere embedded
% create a surface mesh for a 10 mm radius sphere

[nbox,ebox]=meshgrid6(0:60:60,0:60:60,0:60:60);
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
    0.002 1.0 0.01 1.37;
    0.050 5.0 0.9 1.37];
cfg.debuglevel='TP';
cfg.isreflect=0;
cfg.node = nbox;
cfg.elem = ebox;
cfg.outputtype = 'fluence';
%tiny placeholder capsule to avoid crashing due to rigid SBT
cfg.capsulecenters = [59, 59, 59; 59.1, 59.1, 59.1];
cfg.capsulewidths = [0.001];
cfg.spheres = [61 61 61 0.5];
cfg.compute = 'optix';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

mcxplotvol(log(fluence_MMC));

% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(30,1:60,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = 10;
contour(log(fluence_MMC), clines, '-k')
axis equal;