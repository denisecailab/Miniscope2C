%% Simulate full system
LED_Pos = linspace(-1.2,1.2,3);
EWL_Power = 2;
f = [];
figure(1);
clf
for xShift = LED_Pos
    clf
% xShift = -.5;
    nrays = 300;
    rays_Excitation = Rays( nrays, 'source', [ 14.4+xShift 5.35 0], [ 0 -1 0 ], .15, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);
    
    %
    % clf
    f(:,end+1) = ms_v4_Excitation(rays_Excitation, EWL_Power);
    
end

nrays = 3;
clf
for i=1:size(f,2)
    if i == 1
        rays_Excitation = Rays( nrays, 'source', [14.4+LED_Pos(i) 5.4 0], [ 0  -1 0 ], .5, 'hexagonal', 'air',480*10^(-9),[ 0 0 1],1);
    else
        ray = Rays( nrays, 'source', [14.4+LED_Pos(i) 5.4 0], [ 0 -1 0 ], .5, 'hexagonal', 'air',480*10^(-9),[ 0 0 1],1);
        rays_Excitation = rays_Excitation.append(ray);
    end
end
ms_v4_Excitation(rays_Excitation, EWL_Power);
% 
%%
nrays = 5;
for i=1:size(f,2)
    col = [0 0 0]
    col(i) = 1
    if i == 1
        rays_Emission = Rays( nrays, 'source', f(:,i)', [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9), col, 1);
    else
        ray = Rays( nrays, 'source', f(:,i)', [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9), col,1);
        rays_Emission = rays_Emission.append(ray);
    end
end
screen = ms_v4_Emission(rays_Emission,EWL_Power);
% view([0 0 1])
% figure(2)
% imshow(screen.image,[])
% colormap jet
% colorbar
% % view([0 0 1]);
% figure(3)
% plot(f(2,:),f(1,:),'linewidth',2)

%% Find optimal CMOS sensor location
nrays = 100;
EWL_Power = 0;

rays_Emission = Rays( nrays, 'source', [-0.734 0 0], [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);

bench = Bench;
[bench xO xI] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);
[xI xO]
screen = Screen( [xI 0 0 ], 3, 3, 1000, 1000 );
bench.append( screen );

dv = [];
xScreen = linspace(xI-2,xI+2,100);
for pos = xScreen
    screen.r(1) = pos;
    rays_through = bench.trace( rays_Emission );
    [ ~, dv( end+1 ) ] = rays_through( end ).stat;
end
plot(xScreen,dv,'linewidth',2);
[ mdv, mi ] = min( dv );
focal = xScreen(mi)

%% Adjust LED position for correct image length
clf
nrays = 3000;
xShift = 1.4;
rays_Excitation = Rays( nrays, 'source', [ 14.4+xShift 5.35 0], [ 0 -1 0 ], .1, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);
EWL_Power = 0;
 ms_v4_Excitation(rays_Excitation, EWL_Power);