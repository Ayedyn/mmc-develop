%% Plot results from optix IMMC:

% read output
fid=fopen('optixIMMC_fluenceHomog1e7.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[100,100,100,10]);
%sum along the time dimension
res = sum(res, 4);

% visualize
energy=res;
mcxplotvol(log(energy));

% prepare DMMC results
energy=squeeze(energy(50,:,:,:));

energy=rot90(energy);
clines = 10;
contour(log(energy), clines, ':r')
axis equal;

%% Plot results from Yaoshen's version:
%% node-based iMMC, benchmark B2

% Mesh preparation: generate bounding box and insert edge
[nbox,ebox]=meshgrid6(0:1,0:1,0:1);
fbox=volface(ebox);
EPS=0.001;
nbox=[nbox; [1-EPS 0.5 0.5]; [EPS 0.5 0.5]];  % insert new nodes (node 9 and 10)
fbox=[fbox; [9 9 10]];  % insert new edge coneected by node 9 and 10

clear cfg0 cfg;

cfg0.nphoton=1e7;
cfg0.srcpos=[0.5 0.5 1];
cfg0.srcdir=[0 0 -1];
cfg0.prop=[0 0 1 1;0.0458 35.6541 0.9000 1.3700; 0.0458 35.6541 0.9000 1.3700];
cfg0.tstart=0;
cfg0.tend=5e-9;
cfg0.tstep=5e-9;
cfg0.debuglevel='TP';
cfg0.method='grid';
cfg0.steps=[0.01 0.01 0.01];
cfg0.isreflect=1;
cfg0.gpuid=-1;
cfg0.outputtype='fluence';
cfg0.isnormalized=1;

% (a) generate bounding box and insert edge
[nbox,ebox]=meshgrid6(0:1,0:1,0:1);
fbox=volface(ebox);
nbox=[nbox; [0.5 0.5 0.5]];  % insert new nodes (node 9)

cfg=cfg0;

% (b) generate mesh
[cfg.node,cfg.elem]=s2m(nbox,num2cell(fbox,2),1,100,'tetgen1.5',[],[],'-YY');
cfg.elemprop=ones(size(cfg.elem,1),1);

% (c) label the edge that has node 9 and 10 and add radii
cfg.noderoi=zeros(size(cfg.node,1),1);
cfg.noderoi(9)=0.1;

% run node-based iMMC
energy_nimmc=mmclab(cfg);
energy_nimmc=energy_nimmc.data;
energy_nimmc=energy_nimmc(1:100,1:100,1:100);
% visualize
mcxplotvol(log(energy_nimmc));

% contour plot:
% prepare DMMC results
energy_nimmc=squeeze(energy_nimmc(50,:,:,:));

energy_nimmc=rot90(energy_nimmc);
clines = 10;
contour(log(energy_nimmc), clines, '-k')
axis equal;

%% Direct comparison
% prepare optixIMMC results
figure;
clines = 10;-4:0.5:1;
contour(log(energy), clines, ':r')
axis equal;
hold on;

% contour plot:
% prepare IMMC results
clines = 10;
contour(log(energy_nimmc), clines, '-k')
axis equal;

%% Prepare results from regular MMC:
clear cfg0 cfg;
% Mesh preparation: generate bounding box and insert edge
[nbox,ebox]=meshgrid6(0:1,0:1,0:1);
fbox=volface(ebox);

% create a surface mesh for a 0.1 mm radius sphere
[nsph_01,fsph_01]=meshasphere([0.5 0.5 0.5],0.1,0.01);
[nsph_01,fsph_01]=removeisolatednode(nsph_01,fsph_01);

[no,fc]=mergemesh(nsph_01,fsph_01,nbox,fbox);
[no,fc]=removeisolatednode(no,fc);
ISO2MESH_TETGENOPT='-A -q -Y'
[cfg0.node,cfg0.elem]=surf2mesh(no,fc,[0 0 0],[1, 1, 1],1,0.05,[1 1 1;0.5 0.5 0.5]);

cfg0.nphoton=1e8;
cfg0.srcpos=[0.5 0.5 1];
cfg0.srcdir=[0 0 -1];
cfg0.prop=[0 0 1 1;0.0458 35.6541 0.9000 1.3700; 23.0543 9.3985 0.9000 1.3700];
cfg0.tstart=0;
cfg0.tend=5e-9;
cfg0.tstep=5e-9;
cfg0.debuglevel='TP';
cfg0.method='grid';
cfg0.steps=[0.01 0.01 0.01];
cfg0.isreflect=1;
cfg0.outputtype='fluence';
cfg0.isnormalized=1;

% run MMC
fluence=mmclab(cfg0);

fluence=fluence.data;
fluence=fluence(1:100,1:100,1:100);
% visualize
mcxplotvol(log(fluence));

% contour plot:
% prepare DMMC results
fluence=squeeze(fluence(50,:,:,:));

fluence=rot90(fluence);
clines = 10;
contour(log(fluence), clines, '-k')
axis equal;
