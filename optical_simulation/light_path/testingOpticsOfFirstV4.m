%% test objective focual length
% Will plot the working distance of the objective at different EWL powers
EWL_Power_Range = [-5, 0, 13];
c = ['r','g','b'];

nrays = 40;
rays_in = Rays( nrays, 'collimated', [12 0 0], [ -1 0 0 ], 1, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);
EWL_idx = 0;
dv = [];
for EWL_Power = EWL_Power_Range
    EWL_idx = EWL_idx + 1;
    bench = Bench;
    [bench, xO, xI] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);
    % [xI xO]
    bench.elem = bench.elem(1:8);
    bench.elem = flip(bench.elem);
        bench.cnt = length(bench.elem);
        for i=1:bench.cnt
            bench.elem{i}.glass = flip(bench.elem{i}.glass);
        end

    screen = Screen( [-1 0 0 ], 10, 10, 1000, 1000 );
    screen.rotate([0 0 1], pi);
    bench.append( screen );


    % rays_through = bench.trace( rays_in );

    % clf
    %     bench.draw( rays_through,'lines',0.33,1,0);
    %     view([0, 0,1])

    
    xScreen = linspace(-2, -.5,100);
    count = 0;
    for pos = xScreen
        count = count + 1;
        screen.r(1) = pos;
        rays_through = bench.trace( rays_in );
        [ ~, dv(EWL_idx, count ) ] = rays_through( end ).stat;
    end
end
clf
for i=1:length(EWL_Power_Range)
    plot(xScreen,dv(i,:),'linewidth',2,'color',c(i));
    hold on
end
legend(['EWL Power: ' num2str(EWL_Power_Range(1))], ...
    ['EWL Power: ' num2str(EWL_Power_Range(2))], ...
    ['EWL Power: ' num2str(EWL_Power_Range(3))]);
hold off
xlabel('Working Distance')
[ mdv, mi ] = min( dv );
% focal = xScreen(mi)

%% test emision focual length
% Will plot the focal length of emission module

nrays = 100;
rays_in = Rays( nrays, 'collimated', [0 0 0], [ 1 0 0 ], 1, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);
dv = [];
    bench = Bench;
    [bench, xO, xI, endLensPos] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',0,0);
    % [xI xO]
    bench.elem = bench.elem(9:end);
    bench.cnt = length(bench.elem);
    screen = Screen( [20 0 0 ], 10, 10, 1000, 1000 );
    bench.append( screen );


    % rays_through = bench.trace( rays_in );

    % clf
    %     bench.draw( rays_through,'lines',0.33,1,0);
    %     view([0, 0,1])

    
    xScreen = linspace(19, 22,50);
    count = 0;
    for pos = xScreen
        count = count + 1;
        screen.r(1) = pos;
        rays_through = bench.trace( rays_in );
        [ ~, dv( count ) ] = rays_through( end ).stat;
    end
[ mdv, mi ] = min( dv );
clf
    plot(xScreen-endLensPos,dv(:),'linewidth',2);
legend(['Ideal position: ' num2str(xScreen(mi) - endLensPos)]);
xlabel('Ideal CMOS Radial Position From Back of Tube Lens')

focal = xScreen(mi)
%% Find the optimal position of excitation LED for wide field illumination
EWL_Power = 0;

bench = Bench;

[bench xO xI endLensPos] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);
xI
% Remove extra emission and dichroic surfaces and flip order
bench.elem = bench.elem(1:(end-4));
bench.elem = flip(bench.elem);
bench.cnt = length(bench.elem);
for i=1:bench.cnt
    bench.elem{i}.glass = flip(bench.elem{i}.glass);
end


% Add in half ball lens
posBallLens = 1.5+3;

surf{1} = Plane([endLensPos+posBallLens+1.5 0 0], 3,3,{'air' 'bk7'});
surf{2} = Lens([endLensPos+posBallLens 0 0], -3,1.501,0,{'bk7' 'air'});

bench.prepend(surf{2});
bench.prepend(surf{1});
%------------------------------------------------
%Add excitation filter
plane1 = Plane([endLensPos+posBallLens+1.5+.1 0 0], 4,4, {  'bk7' 'air'});

plane2 = Plane([endLensPos+posBallLens+1.5+1.1 0 0], 3,3, { 'air' 'bk7' });

bench.prepend(plane1);
bench.prepend(plane2);
%------------------------------------------------

screen = Screen( [ -.75  0 0 ], 1, 1, 20, 20 );
    screen.rotate([0 0 1], pi);
    bench.append( screen );
    
% figure(1)
nrays = 100;
yShift = 0;
zShift = 0;
LEDPos = 6+3;%5.35
spread = .25;
rays_in = Rays( nrays, 'source', [ endLensPos+LEDPos 0 0], [-1 0 0 ], spread, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);

for yShift = -.5:.25:.5
    
    for zShift = -.5:.25:.5
        if ((yShift ~=0) || (zShift ~= 0))
            rays = Rays( nrays, 'source', [ endLensPos+LEDPos yShift zShift], [ -1  0 0 ], spread, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);
            rays_in = rays_in.append(rays);
        end
    end
end
rays_through = bench.trace( rays_in );
% bench.draw( rays_through, 'lines' );
figure(1)
clf
bench.draw( rays_through,'lines',0.33,1,0);
view([0 0 1]);

figure(5)
clf
s = screen;
pcolor(s.image)
colormap jet
daspect([1 1 1])
colorbar
title(num2str(sum(s.image(:))))
% figure(6)
% a = screen.image;
% hist(a(:),100);
