clear cfg

%% create surface mesh
[no_box,el_box]=meshgrid6(0:60:60,0:60:60,0:60:60);
fc_box=volface(el_box);

%% create volume mesh
ISO2MESH_TETGENOPT='-Y -A';
[cfg.node,cfg.elem]=surf2mesh(no_box,fc_box,[0 0 0],[60 60 60],1.0,100,...
    [1,1,1]);

%% set up cfg
cfg.nphoton=1e8;

cfg.srcpos=[30 30 0.01];
cfg.srcdir=[0 0 1];

cfg.prop=[0.000,  0, 1, 1;
          0.005,  1, 0, 1.37
          0.005,  1, 0, 1.37]; %box

% time-gate
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

% dual grid MC
cfg.method='grid';

% output energy deposition
cfg.outputtype='fluence';

% gpu setting
cfg.gpuid=1;

%% create simulation parameters
%%-----------------------------------------------------------------

%tiny placeholder capsule to avoid crashing due to rigid SBT
cfg.capsulecenters = [59, 59, 59; 59.1, 59.1, 59.1];
cfg.capsulewidths = [0.1];
cfg.spheres = [0 59 59 0.1];
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