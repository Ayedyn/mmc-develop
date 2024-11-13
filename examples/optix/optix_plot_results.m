% read output
fid=fopen('optix.bin');
output=fread(fid,'float64');

% retrieve time-resolved results(10 time gates)
res=reshape(output,[100,100,100,10]);

% convert to cw solution and visualize
cw=sum(res,4);
mcxplotvol(log(cw));