%% Prep a 60 by 60 by 60 mm cube with a 10mm sphere embedded
% create a surface mesh for a 10 mm radius sphere
[nsph1_3,fsph1_3]=meshasphere([20 30 30],3,1.0);
[nsph1_3,fsph1_3]=removeisolatednode(nsph1_3,fsph1_3);

[nsph2_3,fsph2_3]=meshasphere([40 30 30],3,1.0);
[nsph2_3,fsph2_3]=removeisolatednode(nsph2_3,fsph2_3);

[nsph3_3,fsph3_3]=meshasphere([30 30 30],3,1.0);
[nsph3_3,fsph3_3]=removeisolatednode(nsph3_3,fsph3_3);

[nsph4_3,fsph4_3]=meshasphere([20 20 30],3,1.0);
[nsph4_3,fsph4_3]=removeisolatednode(nsph4_3,fsph4_3);

[nsph5_3,fsph5_3]=meshasphere([40 20 30],3,1.0);
[nsph5_3,fsph5_3]=removeisolatednode(nsph5_3,fsph5_3);

[nsph6_3,fsph6_3]=meshasphere([30 20 30],3,1.0);
[nsph6_3,fsph6_3]=removeisolatednode(nsph6_3,fsph6_3);

[nsph7_3,fsph7_3]=meshasphere([20 40 30],3,1.0);
[nsph7_3,fsph7_3]=removeisolatednode(nsph7_3,fsph7_3);

[nsph8_3,fsph8_3]=meshasphere([40 40 30],3,1.0);
[nsph8_3,fsph8_3]=removeisolatednode(nsph8_3,fsph8_3);

[nsph9_3,fsph9_3]=meshasphere([30 40 30],3,1.0);
[nsph9_3,fsph9_3]=removeisolatednode(nsph9_3,fsph9_3);

[nbox,ebox]=meshgrid6(0:60:60,0:60:60,0:60:60);
fbox=volface(ebox);
[no,fc]=mergemesh(nsph1_3,fsph1_3,nsph2_3,fsph2_3,nsph3_3,fsph3_3, ...
  nsph4_3,fsph4_3,nsph5_3,fsph5_3,nsph6_3,fsph6_3,nsph7_3,fsph7_3, ...
  nsph8_3,fsph8_3,nsph9_3,fsph9_3,nbox,fbox);
[no,fc]=removeisolatednode(no,fc);

[node, elem]=surf2mesh(no,fc,[0 0 0],[60 60 60],1,100,[1 1 1;20 30 30;30 30 30;40 30 30;20 20 30;30 20 30;40 20 30;20 40 30;30 40 30;40 40 30]);%thin layer
%% set material of all spheres to 2

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
cfg.node = node;
cfg.elem = elem;
cfg.outputtype = 'fluence';

fluence_MMC = mmclab(cfg);
fluence_MMC = fluence_MMC.data;

mcxplotvol(log(fluence_MMC));

% prepare DMMC results
fluence_MMC=squeeze(fluence_MMC(30,1:60,1:60,:));

fluence_MMC=rot90(fluence_MMC);
clines = 10;
contour(log(fluence_MMC), clines, '-k')
axis equal;