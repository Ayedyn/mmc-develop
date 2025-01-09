% read output
fid=fopen('optix_basic_capsule_1e8_nodebug.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[60,60,60,10]);
%sum along the time dimension
res = sum(res, 4);

% visualize
fluence_IMMC=res;
fluence_IMMC=fluence_IMMC./1e8;

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(30,:,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = -15:1.2:-1;
figure;
contour(log(fluence_IMMC), clines, ':r')
axis equal;

%% Load pre-generated capsule mesh:
load('capsule_mesh.mat', 'no','el');

% create simulation parameters
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
cfg.node = no;
cfg.elem = el;
cfg.outputtype = 'fluence';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

hold on;
% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(30,1:60,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = -15:1:-1;
contour(log(fluence_MMC), clines, '-k')
axis equal;

% plot location of capsule
rectangle('Position',[5 20 50 20],'Curvature',1,'EdgeColor','r','LineWidth',3)

% =============== load result from iMMC ===================
% read output
fid=fopen('optix_basic_capsule_1e8_nodebug.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[60,60,60,10]);
%sum along the time dimension
res = sum(res, 4);

% visualize
fluence_IMMC=res;
fluence_IMMC=fluence_IMMC./1e8;

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(30,:,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = -15:1:-1;
contour(log(fluence_IMMC), clines, ':r')
axis equal;

legend('MMC', 'iMMC')

%% Load pre-generated capsule mesh:
load('capsule_mesh.mat', 'no','el');

% create simulation parameters and plot axial view
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
cfg.node = no;
cfg.elem = el;
cfg.outputtype = 'fluence';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

figure;
hold on;

% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(1:60,30,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = -15:1:-1;
contour(log(fluence_MMC), clines, '-k')
axis equal;

% =============== load result from iMMC ===================
% read output
fid=fopen('optix_basic_capsule_1e8_nodebug.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[60,60,60,10]);
%sum along the time dimension
res = sum(res, 4);

% visualize
fluence_IMMC=res;
fluence_IMMC=fluence_IMMC./1e8;

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(:,30,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = -15:1:-1;
contour(log(fluence_IMMC), clines, ':r')
axis equal;

legend('MMC', 'iMMC')

viscircles([30,30],10)