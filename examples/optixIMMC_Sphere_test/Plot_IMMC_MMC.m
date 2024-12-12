% read output
fid=fopen('optix_basic_sphere_1e8_fluence.bin');
output=fread(fid,'float64');

% retrieve results
res=reshape(output,[60,60,60,10]);
%sum along the time dimension
res = sum(res, 4);

% visualize
fluence_IMMC=res;
fluence_IMMC=fluence_IMMC./1e7;
mcxplotvol(log(fluence_IMMC));

% prepare DMMC results
fluence_IMMC=squeeze(fluence_IMMC(30,:,:,:));

fluence_IMMC=rot90(fluence_IMMC);
clines = 10;
contour(log(fluence_IMMC), clines, ':r')
axis equal;

% plot circle for where sphere is embedded
viscircles([30,30],10)




