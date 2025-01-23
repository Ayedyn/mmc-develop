% read output
fid=fopen('optix.bin');
output=fread(fid,'float64');

% retrieve spatial results
res=reshape(output,[60,60,60,10]);

% visualize
fluence_IMMC=sum(res,4);
fluence_IMMC=fluence_IMMC./1e8;

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(30,:,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = -20:0.2:-1;
figure;
contour(log(fluence_IMMC), clines, ':r')