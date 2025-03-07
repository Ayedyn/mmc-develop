%% Prep a 60 by 60 by 60 mm cube with a 10mm Radius 30mm long capsule embedded

% create a surface mesh for the first 10 mm radius sphere
[nsph1_10,fsph1_10, el1_10]=meshasphere([30 15 30],10,1.0);

% create a surface mesh for the second 10 mm radius sphere
[nsph2_10,fsph2_10, el2_10]=meshasphere([30 45 30],10,1.0);

% create a surface mesh for a cylinder going to the spheres
[ncyl, fcyl, ecyl] = meshacylinder([30 15 30], [30 45 30], 10, 1);
ecyl = ecyl(:,1:4);

% use boolean operations to combine the capsule surfaces
[newnode,newface]=surfboolean(ncyl,fcyl,'second',nsph1_10,fsph1_10);
[newnode,newface]=surfboolean(newnode,newface,'second',nsph2_10,fsph2_10);

% create a cube surface to hold the capsule
[nbox,ebox]=meshgrid6(0:60:60,0:60:60,0:60:60);
fbox=volface(ebox);
ebox(:,5)=1;

[newnode, newface] = mergemesh(nbox, fbox, newnode, newface);

[no, el]=surf2mesh(newnode, newface,[0,0,0],[60,60,60], 1, 1000, [1,1,1; 30, 15, 30; 30, 30, 30; 30, 45, 30]);

% relabel all property values of the capsule
capsule_material_indices = el(:, 5)>1;
el(capsule_material_indices, 5)=2;
%% Load pre-generated capsule mesh:
load('capsule_mesh.mat', 'no','el');

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
    0.050 5.0 0.9 1.37;
    0.002 1.0 0.01 1.37];
cfg.debuglevel='TP';
cfg.isreflect=0;
cfg.node = no;
cfg.elem = el;
cfg.outputtype = 'fluence';
%cfg.compute = 'optix';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

mcxplotvol(log(fluence_MMC));

% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(30,1:60,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = -15:1.2:-1;
contour(log(fluence_MMC), clines, '-k')
axis equal;

% plot location of capsule
rectangle('Position',[5 20 50 20],'Curvature',1,'EdgeColor','r','LineWidth',3)


