function f = ms_v4_Excitation(rays_in,EWL_Power)
%MS_V4_OPTICS Summary of this function goes here
%   Detailed explanation goes here
    bench = Bench;
    
    [bench xO xI] = buildOpticStack(bench,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);
    xI
    % Remove extra emission surfaces and flip order
    bench.elem = bench.elem(1:(end-3));
    bench.elem = flip(bench.elem);
    bench.cnt = length(bench.elem);
    for i=1:bench.cnt
        bench.elem{i}.glass = flip(bench.elem{i}.glass);
    end
    bench.elem{1}.glass = {'air' 'mirror' };
    %------------------------------------------------
    %Add excitation filter
    plane1 = Plane([14.4 3 0], 4,4, {  'bk7' 'air'});
    plane1.rotate([0 0 1],-pi/2);
    plane2 = Plane([14.4 4 0], 3,3, { 'air' 'bk7' });
    plane2.rotate([0 0 1],-pi/2);
    bench.prepend(plane1);
    bench.prepend(plane2);
    %------------------------------------------------
    
    screen = Screen( [ -2  0 0 ], 4, 4, 1024, 1024 );
    screen.rotate([0 0 1], pi);
    bench.append( screen );

    rays_through = bench.trace( rays_in );
    [f, ~] = rays_through(end-1).focal_point();
    display(f);
    
%     b.draw( rays, draw_fl, alpha, scale, new_figure_fl )
    bench.draw( rays_through,'lines',0.33,1,0);  
%     hold on
%     plot3(f(1), f(2), f(3),'or','markersize',20)
%     hold off
%     view([0 0 1]);
%     daspect([1 1 1]);
%         figure(2)
%     imshow( screen.image, [] );
end

