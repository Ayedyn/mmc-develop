fid=fopen('optix.bin');% read output
output=fread(fid,'float64');

% retrieve time-resolved results(10 time gates)
res=reshape(output,[61,61,61,1]);

% convert to cw solution and visualize
cw=sum(res,4);
mcxplotvol(log(cw));

% visualize
fluence_IMMC=cw;
fluence_IMMC=fluence_IMMC./1e8;

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(30,:,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = -15:0.5:-1;
contour(log(fluence_IMMC), clines, ':r')
axis equal;