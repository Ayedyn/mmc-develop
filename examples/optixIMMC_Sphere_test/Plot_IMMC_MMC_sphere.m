% read output
fid=fopen('optix_basic_sphere_1e8.bin');
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
contour(log(fluence_IMMC), clines, ':r')
axis equal;

%% Plot both MMC and iMMC
% read output
fid=fopen('optix_basic_sphere_1e8.bin');
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

figure;

fluence_IMMC=rot90(fluence_IMMC);
clines = -15:1.2:-1;
contour(log(fluence_IMMC), clines, ':r')
axis equal;
hold on;
% Prep a 60 by 60 by 60 mm cube with a 10mm sphere embedded
% create a surface mesh for a 10 mm radius sphere
[nsph_10,fsph_10]=meshasphere([30 30 30],10,1.0);
[nsph_10,fsph_10]=removeisolatednode(nsph_10,fsph_10);

[nbox,ebox]=meshgrid6(0:60:60,0:60:60,0:60:60);
fbox=volface(ebox);
[no,fc]=mergemesh(nsph_10,fsph_10,nbox,fbox);
[no,fc]=removeisolatednode(no,fc);

[node, elem]=surf2mesh(no,fc,[0 0 0],[60 60 60],1,100,[1 1 1;30 30 30]);%thin layer

% create simulation parameters
%-----------------------------------------------------------------
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
cfg.node = node;
cfg.elem = elem;
cfg.outputtype = 'fluence';
cfg.compute = 'optix';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(30,1:60,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = -15:1.2:-1;
contour(log(fluence_MMC), clines, '-k')
axis equal;

legend('iMMC', 'MMC')

% plot circle for where sphere is embedded
viscircles([30,30],10)




