%{
cfg.vol=zeros(64,64,64);
cfg.vol(21:39,21:39,5:end)=1;
cfg.prop=[0.001,0.01,0.999,1.0; 0.01, 1.0, 0.01, 1.05];
cfg.srcpos=[32,32,0.01];
cfg.srcdir=[0 0 1];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=cfg.tend;
cfg.nphoton=2^16;

flux=mcxlab(cfg);
%}

cfg.vol=ones(64,64,64);
% cfg.vol(1:end,1:end,1:end)=0;
% cfg.vol(22:40,22:40,34:end)=1;
cfg.vol(30:40,30:40,5:15)=1;
cfg.prop=[0, 0, 1, 1.0; 0.05, 1.0, 0.0, 1.0; 1, 1, 0.0, 1.0];
cfg.srcpos=[32.0,32.0,0.1]+[1, 1, 1];
cfg.srcdir=[0 0 1];
cfg.tstart=0;
cfg.tend=5e-9;
cfg.tstep=cfg.tend;
cfg.nphoton=2^21;
cfg.outputtype='energy';
cfg.isnormalized=0;
cfg.isreflect=0;

%mcx2json(cfg, 'cube64');
flux2=mcxlab(cfg);