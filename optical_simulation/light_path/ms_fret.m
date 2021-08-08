ewlPower = 0;
pos_start = 0;
xO = -1; 

%% build optics
bench_emm1 = Bench;
bench_emm2 = Bench;
bench_ext = Bench;

% Achr 45-089, 6mm FL
pos_lens1 = pos_start;
[lens1, CT_lens1, ~] = buildAchrLens('achr_45089',1,pos_start); 
bench_emm1.append(lens1{1});
bench_emm1.append(lens1{2});
bench_emm1.append(lens1{3});
bench_emm2.append(lens1{1});
bench_emm2.append(lens1{2});
bench_emm2.append(lens1{3});
bench_ext.append(lens1{1});
bench_ext.append(lens1{2});
bench_ext.append(lens1{3});

% Achr 45-089, 6mm FL
pos_lens2 = pos_lens1 + CT_lens1 + 0.1;
[lens2, CT_lens2, ~] = buildAchrLens('achr_45089',1,pos_lens2); 
bench_emm1.append(lens2{1});
bench_emm1.append(lens2{2});
bench_emm1.append(lens2{3});
bench_emm2.append(lens2{1});
bench_emm2.append(lens2{2});
bench_emm2.append(lens2{3});
bench_ext.append(lens2{1});
bench_ext.append(lens2{2});
bench_ext.append(lens2{3});

% EWL 
pos_ewl = pos_lens2 + CT_lens2 + 0.1;
CT_ewl = 1.7;
ewl{1} = Plane([pos_ewl 0 0], 2.6, { 'air' 'bk7' } );  
if ewlPower == 0
    ewl{2} = Plane([pos_ewl+CT_ewl 0 0], 2.6, {'bk7' 'air'} ); 
else
    n = 1.523; %N-BK7
    r = (n-1)*1000/ewlPower;
    ewl{2} = Lens([pos_ewl+CT_ewl 0 0], 2.6, -1*r, 0, {'bk7' 'air'});
end
bench_emm1.append(ewl{1});
bench_emm1.append(ewl{2});
bench_emm2.append(ewl{1});
bench_emm2.append(ewl{2});
bench_ext.append(ewl{1});
bench_ext.append(ewl{2});

% led screen
% pos_led = pos_ewl + CT_ewl + 0.01;
% screen_led = Screen([pos_led 0 0], 3, 3, 1000, 1000);
% bench_ext.append(screen_led);

% Dichroic Mirror 1
pos_dich1 = pos_ewl + CT_ewl + 3.5;
dich1{1} = Plane([pos_dich1 0 0], 6,4, { 'air' 'bk7' } ); 
dich1{1}.rotate([0 0 1], pi/4);
dich1{2} = Plane([pos_dich1+sqrt(1/2) 1*sqrt(1/2) 0], ...
    6,4, {  'bk7' 'air'} ); 
dich1{2}.rotate([0 0 1], pi/4);
dich1{3} = Plane([pos_dich1 0 0], 6,4, { 'air' 'mirror' } ); 
dich1{3}.rotate([0 0 1], pi/4);
bench_emm1.append(dich1{1});
bench_emm1.append(dich1{2});
bench_emm2.append(dich1{1});
bench_emm2.append(dich1{2});
bench_ext.append(dich1{3});

% half-ball lens
pos_ball = 0;
ball{1} = Lens([pos_dich1 pos_ball 0], -3, 1.501, 0, {'air' 'bk7'});
ball{1}.rotate([0 0 1], -pi/2);
ball{2} = Plane([pos_dich1 pos_ball-1.5 0], 3, 3, {'bk7' 'air'});
ball{2}.rotate([0 0 1], -pi/2);
bench_ext.append(ball{1});
bench_ext.append(ball{2});

% excitation filter
pos_ftext = pos_ball - 1.5;
ftext{1} = Plane([pos_dich1 pos_ftext 0], 3,3, { 'air' 'bk7' } );
ftext{1}.rotate([0 0 1], -pi/2);
ftext{2} = Plane([pos_dich1 pos_ftext-1 0], 3,3, {  'bk7' 'air'} );
ftext{2}.rotate([0 0 1], -pi/2);
bench_ext.append(ftext{1});
bench_ext.append(ftext{2});

% Achr 63-691, 10mm FL
pos_lens3 = pos_dich1 + 3;
[lens3, CT_lens3, BFL_lens3] = buildAchrLens('achr_63691',0,pos_lens3); 
bench_emm1.append(lens3{1});
bench_emm1.append(lens3{2});
bench_emm1.append(lens3{3});
bench_emm2.append(lens3{1});
bench_emm2.append(lens3{2});
bench_emm2.append(lens3{3});
xI1 = pos_lens3 + CT_lens3 + BFL_lens3;

% Dichroic Mirror 2
pos_dich2 = pos_lens3 + CT_lens3 + 3*sqrt(1/2) + 0.1;
dich2{1} = Plane([pos_dich2 1*sqrt(1/2) 0], 6,4, { 'air' 'bk7' } );
dich2{1}.rotate([0 0 1], -pi/4);
dich2{2} = Plane([pos_dich2+sqrt(1/2) 0 0], ...
    6,4, {  'bk7' 'air'} );
dich2{2}.rotate([0 0 1], -pi/4);
dich2{3} = Plane([pos_dich2 1*sqrt(1/2) 0], 6,4, { 'air' 'mirror' } );
dich2{3}.rotate([0 0 1], -pi/4);
bench_emm1.append(dich2{1});
bench_emm1.append(dich2{2});
bench_emm2.append(dich2{3});

% Emission Filter A
pos_ftemm1 = xI1 - 1.5;
ftemm1{1} = Plane([pos_ftemm1 0 0], 3,3, { 'air' 'bk7' } ); 
ftemm1{2} = Plane([pos_ftemm1+1 0 0], 3,3, {  'bk7' 'air'} ); 
bench_emm1.append(ftemm1{1});
bench_emm1.append(ftemm1{2});

% screen A
screen1 = Screen( [ xI1  0 0 ], 3, 3, 1000, 1000 );
bench_emm1.append(screen1);

% Emission Filter B
xI2 = xI1 + sqrt(1/2);
pos_ftemm2 = pos_dich2;
ftemm2{1} = Plane([pos_ftemm2 xI2-pos_ftemm2-1.5 0], 3,3, { 'air' 'bk7' } );
ftemm2{1}.rotate([0 0 1], pi/2);
ftemm2{2} = Plane([pos_ftemm2 xI2-pos_ftemm2-0.5 0], 3,3, {  'bk7' 'air'} );
ftemm2{2}.rotate([0 0 1], pi/2);
bench_emm2.append(ftemm2{1});
bench_emm2.append(ftemm2{2});

% screen B
screen2 = Screen( [pos_ftemm2 xI2-pos_ftemm2 0], 3, 3, 1000, 1000 );
screen2.rotate([0 0 1], pi/2);
bench_emm2.append(screen2);


%% draw optics
ray = Rays(19, 'source', [xO 0 0], [ 1 0 0 ], ... 
    0.3, 'hexagonal', 'air', 525*10^(-9), [0 1 0],  1);
ray_col = Rays(19, 'collimated', [xO 0 0], [ 1 0 0 ], ... 
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0],  1);
ray_th1 = bench_emm1.trace(ray);
ray_th2 = bench_emm2.trace(ray);
ray_th3 = bench_ext.trace(ray_col);
bench_emm1.draw(ray_th1, 'lines');
view([0, 0, 1]);
bench_emm2.draw(ray_th2, 'lines');
view([0, 0, 1]);
bench_ext.draw(ray_th3, 'lines');
view([0, 0, 1]);

%% rebuild excitation path and simulate different shifts

% reverse pathway order
bench_ext.elem = flip(bench_ext.elem);
for i=1:bench_ext.cnt
    gls = bench_ext.elem{i}.glass;
    if ~any(strcmp(gls, 'mirror'))
        bench_ext.elem{i}.glass = flip(gls);
    end
end

% screen for excitation
screen_ext = Screen([xO 0 0], 1, 1, 50, 50);
screen_ext.rotate([0 0 1], pi);
bench_ext.append(screen_ext);

% build rays
sh_pos = linspace(-.5, .5, 11);
cols = brewermap(12, 'Paired');
nrays = 1000;
for i = 1:size(sh_pos, 2)
    for j = 1:size(sh_pos, 2)
        ray = Rays(nrays, 'source', ... 
            [pos_dich1+sh_pos(i), 0, sh_pos(j)], [0 1 0], ...
            0.5, 'hexagonal', 'air', 525*10^(-9), cols(i, :),  1);
        try
            rays_ext = rays_ext.append(ray);
        catch err
            rays_ext = ray;
        end
    end
end

% compute deviation on different shifts
pos_org_ball1 = ball{1}.r(2);
pos_org_ball2 = ball{2}.r(2);
pos_org_ftext1 = ftext{1}.r(2);
pos_org_ftext2 = ftext{2}.r(2);
n_ball_sh = 40;
n_led_sh = 50;
ball_shifts = linspace(0, -4, n_ball_sh);
shifts = linspace(0, -5, n_led_sh);
led_loc = [];
led_dv = [];
ims = zeros(n_ball_sh, n_led_sh, 49, 49);

for i=1:n_ball_sh
    ball_sh = ball_shifts(i);
    ball{1}.r(2) = pos_org_ball1 + ball_sh;
    ball{2}.r(2) = pos_org_ball2 + ball_sh;
    ftext{1}.r(2) = pos_org_ftext1 + ball_sh; 
    ftext{2}.r(2) = pos_org_ftext2 + ball_sh;
    dv_led = [];
    for j=1:n_led_sh
        sh = shifts(j);
        rays_ext.r(:, 2) = pos_org_ftext2+ball_sh+sh;
        rays_th = bench_ext.trace(rays_ext);
        sc_im = screen_ext.image(2:end, 2:end);
        sc_im = (sc_im - min(sc_im(:)))/(max(sc_im(:)) - min(sc_im(:)));
        qh = quantile(sc_im(:), 0.95);
        ql = quantile(sc_im(:), 0.05);
        d = (qh - ql) / mean(sc_im(:));
        ims(i, j, :, :) = sc_im;
        dv_led(end+1) = d;
    end
    led_dv = [led_dv; dv_led];
    size(led_dv)
end

%% determine best ball shifts and led positions
imagesc(led_dv);
colorbar;
xlabel('led shifts (0.1 mm)');
ylabel('half ball lens shifts (0.1 mm)');
[~, min_sh] = min(led_dv(:));
[bsh, lsh] = ind2sub(size(led_dv), min_sh);
ball_sh = ball_shifts(bsh);
pos_led = pos_org_ftext2 + ball_sh + shifts(lsh);
ball{1}.r(2) = pos_org_ball1 + ball_sh;
ball{2}.r(2) = pos_org_ball2 + ball_sh;
ftext{1}.r(2) = pos_org_ftext1 + ball_sh;
ftext{2}.r(2) = pos_org_ftext2 + ball_sh;

%% test excitation pathway
sh_pos = linspace(-.5, .5, 11);
cols = brewermap(12, 'Paired');
nrays = 1000;
if exist('rays_ext', 'var') == 1
    clear rays_ext
end
for i = 1:size(sh_pos, 2)
    for j = 1:size(sh_pos, 2)
        ray = Rays(nrays, 'source', ... 
            [pos_dich1+sh_pos(i), pos_led, sh_pos(j)], [0 1 0], ...
            0.5, 'hexagonal', 'air', 525*10^(-9), [0 1 0],  1);
        try
            rays_ext = rays_ext.append(ray);
        catch err
            rays_ext = ray;
        end
    end
end
rays_th_ext = bench_ext.trace(rays_ext);
sc_im = screen_ext.image(2:end, 2:end);
sc_im = (sc_im - min(sc_im(:)))/(max(sc_im(:)) - min(sc_im(:))) * 99;
HeatMap(sc_im, 'Annotate', true, 'Colormap', 'parula');
% bench_ext.draw(rays_th_ext, 'lines');
% view([0 0 1])

%% find y shift of lens
y_sh = linspace(0.3, 0.35, 50);
test_ray1 = Rays(500, 'source', [xO -0.5 0], [1 0 0], ...
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
test_ray2 = Rays(500, 'source', [xO 0.5 0], [1 0 0], ...
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
diffs = [];
for sh = y_sh
    lens3{1}.r(2) = sh;
    lens3{2}.r(2) = sh;
    lens3{3}.r(2) = sh;
    r_th1 = bench_emm1.trace(test_ray1);
    r_th2 = bench_emm1.trace(test_ray2);
    r1s = r_th1(end).r(:,2);
    r2s = r_th2(end).r(:,2);
    r1s(r1s==Inf) = NaN;
    r2s(r2s==Inf) = NaN;
    c1 = abs(nanmean(r1s));
    c2 = abs(nanmean(r2s));
    diffs(end+1) = abs(c1 - c2);
end
plot(y_sh, diffs, 'linewidth', 2);
[~, min_sh] = min(diffs);
sh = y_sh(min_sh);
lens3{1}.r(2) = sh;
lens3{2}.r(2) = sh;
lens3{3}.r(2) = sh;

 %% find x shift of screen2
x_sh = linspace(-0.35, -0.4, 50);
test_ray1 = Rays(500, 'source', [xO -0.5 0], [1 0 0], ...
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
test_ray2 = Rays(500, 'source', [xO 0.5 0], [1 0 0], ...
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
diffs = [];
sc2_pos_org = screen2.r(1);
for sh = x_sh
    screen2.r(1) = sc2_pos_org + sh;
    ftemm2{1}.r(1) = sc2_pos_org + sh;
    ftemm2{2}.r(1) = sc2_pos_org + sh;
    r_th1 = bench_emm2.trace(test_ray1);
    r_th2 = bench_emm2.trace(test_ray2);
    r1s = r_th1(end).r(:,1);
    r2s = r_th2(end).r(:,1);
    r1s(r1s==Inf) = NaN;
    r2s(r2s==Inf) = NaN;
    c1 = abs(nanmean(r1s) - screen2.r(1));
    c2 = abs(nanmean(r2s) - screen2.r(1));
    diffs(end+1) = abs(c1 - c2);
end
plot(x_sh, diffs, 'linewidth', 2);
[~, min_sh] = min(diffs);
sh = x_sh(min_sh);
screen2.r(1) = sc2_pos_org + sh;
ftemm2{1}.r(1) = sc2_pos_org + sh;
ftemm2{2}.r(1) = sc2_pos_org + sh;

%% find optimal screen location
test_rays = Rays(500, 'source', [xO 0 0], [ 1 0 0 ], ... 
        0.2, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
shifts = linspace(-.5, -1.5, 100);
dv_sc1 = [];
dv_sc2 = [];
pos_org_sc1 = screen1.r(1);
pos_org_sc2 = screen2.r(2);
for sh = shifts
    screen1.r(1) = pos_org_sc1 + sh;
    rays_th1 = bench_emm1.trace(test_rays, 1);
    [~, d, ~] = rays_th1(end).stat;
    dv_sc1(end+1) = d;
    screen2.r(2) = pos_org_sc2 + sh;
    rays_th2 = bench_emm2.trace(test_rays, 1);
    [~, d, ~] = rays_th2(end).stat;
    dv_sc2(end+1) = d;
end
[~, min_dv1] = min(dv_sc1);
[~, min_dv2] = min(dv_sc2);
screen1.r(1) = pos_org_sc1 + shifts(min_dv1);
ftemm1{1}.r(1) = ftemm1{1}.r(1) + shifts(min_dv1);
ftemm1{2}.r(1) = ftemm1{2}.r(1) + shifts(min_dv1);
screen2.r(2) = pos_org_sc2 + shifts(min_dv2);
ftemm2{1}.r(2) = ftemm2{1}.r(2) + shifts(min_dv2);
ftemm2{2}.r(2) = ftemm2{2}.r(2) + shifts(min_dv2);
plot(shifts, dv_sc1, shifts, dv_sc2, 'linewidth', 2);
xlabel('shifts');
ylabel('std');
legend({'pass', 'reflect'});
diff = abs(shifts(min_dv1) - shifts(min_dv2))

%% draw optics
ray = Rays(19, 'source', [xO 0 0], [ 1 0 0 ], ... 
    1, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
ray_th1 = bench_emm1.trace(ray);
ray_th2 = bench_emm2.trace(ray);
bench_emm1.draw(ray_th1, 'lines');
view([0, 0, 1]);
bench_emm2.draw(ray_th2, 'lines');
view([0, 0, 1]);

%% build rays
ypos = linspace(-0.5, 0.5, 11);
cols = brewermap(12, 'Paired');
nrays = 500;
x_sh = linspace(0, 0.2, 40);
for i = 1:size(ypos, 2)
    ray = Rays(nrays, 'source', [xO ypos(i) 0], [ 1 0 0 ], ... 
        1, 'hexagonal', 'air', 525*10^(-9), cols(i, :), 1);
    dv_sh = [];
    for sh = x_sh
        ray.r(:, 1) = xO + sh;
        r_th = bench_emm1.trace(ray);
        [~, d, ~] = r_th(end).stat;
        dv_sh(end+1) = d;
    end
    [~, min_sh] = min(dv_sh);
    ray.r(:, 1) = xO + x_sh(min_sh);
    try
        rays = rays.append(ray);
    catch err
        rays = ray;
    end 
end

%% trace rays
rays_th1 = bench_emm1.trace(rays);
rays_th2 = bench_emm2.trace(rays);

%% draw rays
bench_emm1.draw(rays_th1, 'lines');
view([0, 0, 1])
bench_emm2.draw(rays_th2, 'lines');
view([0, 0, 1])

%% draw image
subplot_tight(2, 1, 1);
imshow(screen1.image);
subplot_tight(2, 1, 2);
imshow(screen2.image);

%% calculate effective aperture
angs = linspace(-pi/2, pi/2, 180);
tans = tan(angs);
ang_lb = [];
for t = tans
    test_ray = Rays(5, 'source', [xO 0.5 0], [1 t 0], ...
        0, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);
    r_th = bench_emm1.trace(test_ray);
    if all(all(isnan(r_th(end).r)))
        ang_lb(end+1) = 0;
    else
        ang_lb(end+1) = 1;
    end
end
ang_ft = angs(ang_lb > 0);
ang_max = max(ang_ft);

%% plot effective apperture
test_ray1 = Rays(7, 'source', [xO -0.5 0], [1 tan(-ang_max) 0], ...
    0.2, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);

test_ray2 = Rays(7, 'source', [xO 0.5 0], [1 tan(ang_max) 0], ...
    0.2, 'hexagonal', 'air', 525*10^(-9), [0 1 0], 1);

test_rays = test_ray1.append(test_ray2);
test_th = bench_emm1.trace(test_rays);
bench_emm1.draw(test_th, 'lines');
view([0, 0, 1])
