clear cfg

%% create surface mesh
[no_box,el_box]=meshgrid6(0:1:1,0:1:1,0:1:1);
fc_box=volface(el_box);

[no_sphere,fc_sphere]=meshasphere([0.5 0.5 0.5],0.1,0.01);
[no,fc]=mergemesh(no_box,fc_box,no_sphere,fc_sphere);

%% create volume mesh
ISO2MESH_TETGENOPT='-Y -A';
[cfg.node,cfg.elem]=surf2mesh(no,fc,[0 0 0],[1 1 1],1.0,100,...
    [1,1,1;0.5,0.5,0.5]);

%% set up cfg
cfg.nphoton=1e8;

cfg.srcpos=[0.5 0.5 1];
cfg.srcdir=[0 0 -1];

cfg.prop=[0.000,  0, 1, 1;
          0.0458,  35.6541, 0.9, 1.37;  %box
          23.0543, 9.3985, 0.9, 1.37]; %sphere

% time-gate
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=5e-10;

% dual grid MC
cfg.method='grid';
cfg.steps = [0.01,0.01,0.01];


% output energy deposition
cfg.outputtype='fluence';

% gpu setting
cfg.gpuid=1;

% save configuration
mmc2json(cfg,'optix');