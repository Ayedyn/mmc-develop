function cork_repro(tag)
% Reproduce the skinvessel_shallowvessel meshing step (Cork surfboolean +
% tetgen s2m) on the current platform, for several Cork seeds and for the
% no-Cork variant (cylinder ends inside the box, surfaces never touch).
addpath(genpath(fullfile(pwd, 'iso2mesh')));
outroot = fullfile(pwd, 'results', tag);
if (~exist(outroot, 'dir')), mkdir(outroot); end
sfid = fopen(fullfile(outroot, 'summary.txt'), 'w');

[~, un] = system('uname -a');
say('tag: %s\nmatlab: %s  arch: %s\nuname: %s', tag, version, computer('arch'), un);
say('exe suffix used by iso2mesh: %s\n', getexeext);
bindir = fullfile(pwd, 'iso2mesh', 'bin');
for b = {'cork', 'tetgen'}
    [~, s] = system(sprintf('file "%s"; shasum "%s"', ...
        fullfile(bindir, [b{1} getexeext]), fullfile(bindir, [b{1} getexeext])));
    say('%s', s);
end
tdir = fileparts(mwpath('x'));
say('iso2mesh temp dir: %s\n\n', tdir);

cases = {'default', [], 'cork'; 'seed1', 1, 'cork'; 'seed2', 2, 'cork'; ...
         'seed12345', 12345, 'cork'; 'seed99999', 99999, 'cork'; 'nocork', [], 'nocork'};

for k = 1:size(cases, 1)
    name = cases{k, 1};
    cdir = fullfile(outroot, name);
    mkdir(cdir);
    delete(fullfile(tdir, '*'));
    evalin('base', 'clear ISO2MESH_RANDSEED');
    if (~isempty(cases{k, 2}))
        assignin('base', 'ISO2MESH_RANDSEED', cases{k, 2});
    end

    [no, fc] = latticegrid([0 200], [0 200], [0 10 200]);
    no(end, :) = no(end, :) + 1e-5;
    fc2 = cell2mat(fc);
    fc = [fc2(:, [1 2 3]); fc2(:, [1 3 4])];
    if (strcmp(cases{k, 3}, 'cork'))
        [ncy, fcy] = meshacylinder([-1, 99.5, 50.5], [201, 99.5, 50.5], 20, 8);
        try
            [newnode, newelem] = surfboolean(no, fc, 'first', ncy, fcy);
        catch err
            say('[%s] CORK FAILED: %s\n', name, err.message);
            copyfile(fullfile(tdir, '*'), cdir);
            continue;
        end
    else
        [ncy, fcy] = meshacylinder([1, 99.5, 50.5], [199, 99.5, 50.5], 20, 8);
        newnode = [no; ncy];
        newelem = [fc; fcy + size(no, 1)];
    end
    newelem = newelem(:, 1:3);
    % smallest triangle area and number of near-degenerate (sliver) faces
    e1 = newnode(newelem(:, 2), :) - newnode(newelem(:, 1), :);
    e2 = newnode(newelem(:, 3), :) - newnode(newelem(:, 1), :);
    area = sqrt(sum(cross(e1, e2, 2).^2, 2)) / 2;
    saveoff(newnode, newelem, fullfile(cdir, 'merged_surface.off'));

    seeds = [ones(3, 2) * 100, [5, 150, 50]'];
    try
        [node, elem] = s2m(newnode, newelem, 1, 30, 'tetgen', seeds, []);
        ev = elemvolume(node, elem(:, 1:4));
        say('[%s] PASS  surf: %d nodes %d faces, min area %.3g  | vol: %d nodes %d elems, labels [%s], zero-vol %d\n', ...
            name, size(newnode, 1), size(newelem, 1), min(area), size(node, 1), size(elem, 1), ...
            num2str(unique(elem(:, 5))'), sum(ev == 0));
    catch err
        msg = regexprep(err.message, '\s+', ' ');
        say('[%s] FAIL  surf: %d nodes %d faces, min area %.3g  | %s\n', ...
            name, size(newnode, 1), size(newelem, 1), min(area), msg);
    end
    copyfile(fullfile(tdir, '*'), cdir);
    delete(fullfile(cdir, 'post_vmesh.1.*')); % volume mesh outputs are large and not needed
end
fclose(sfid);

    function say(varargin)
        fprintf(1, varargin{:});
        fprintf(sfid, varargin{:});
    end
end
