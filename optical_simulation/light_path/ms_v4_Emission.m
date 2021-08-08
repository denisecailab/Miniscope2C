function screen = ms_v4_Emission(rays_in,EWL_Power)
%MS_V4_OPTICS Summary of this function goes here
%   Detailed explanation goes here
    bench = Bench;
    
    % [bench xO xI] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);
    [bench xO xI] = buildOpticStack(bench,'fret',EWL_Power,0);
    xI
    xO
    screen = Screen( [ xI-.1  -.3 0 ], 3, 3, 1000, 1000 );
    bench.append( screen );

%     nrays = 20;
%     rays_in = Rays( nrays, 'source', [ xO .4 0], [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);

    rays_through = bench.trace( rays_in );
%     [f ff] = rays_through(end-1).focal_point();
%     f

%     bench.draw( rays_through,'lines',0.33,1,0);
    bench.draw( rays_through,'lines');  
    view([0, 0,1])

%     bench.draw( rays_through,'lines' );  
%     hold on
%     plot3(f(1), f(2), f(3),'or','markersize',20)
%     hold off
%     view([0 0 1]);
%     daspect([1 1 1]);
end

