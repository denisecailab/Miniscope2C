function focal_point = testFieldCurvature()
%TESTSINGLELENS Summary of this function goes here
%   Detailed explanation goes here

    bench = Bench;
    
    bench = buildOpticStack(bench,'Achr9_Achr6_EWL_Achr10',0,0);
     % aspheric 65-567, ~2.7mm FL
%     pos1 = 0;
    
%     %AsphericLens( r, D, R, k, avec, glass )
%     a = [0 ...
%         2.283443987985E-3 ...
%         1.732258175861E-5 ...
%         -1.277960734687E-5 ...
%         -3.711789658434E-6];
% 
%     lens11 = AsphericLens( [pos1 0 0 ], 4, 1/4.623091887902032700E-001, -6.979185555E-1, a, {  'air' 'D-ZLaF52LA' } );
%   
%     lens12 = Plane( [ pos1+1.43 0 0 ], 4, { 'D-ZLaF52LA' 'air' } );
%     bench.append(lens11);
%     bench.append(lens12);
%     %--------------------------------
    
%     %tube lens achromat 63691 FL = 10mm
%     pos3= 0;
%     lens31 = Lens([pos3+0 0 0], 4,7.12,0,{'air' 'baf10'});
%     lens32 = Lens([pos3+2 0 0], 4,-4.22,0,{'baf10' 'sf10'});
%     lens33 = Lens([pos3+3 0 0], 4,-33.66,0,{'sf10' 'air'});
%     bench.append( lens31 );
%     bench.append( lens32 );
%     bench.append( lens33 );
%     %----------------------------------

%     %achromat 65567 FL = 3mm
%     pos3= 0;
%     lens31 = Lens([pos3+0 0 0], 2,1.68,0,{'air' 's-phm52'});
%     lens32 = Lens([pos3+1 0 0], 2,-1.68,0,{'s-phm52' 'lasfn9'});
%     lens33 = Lens([pos3+2 0 0], 2,-7.39,0,{'lasfn9' 'air'});
% 
%     bench.append( lens31 );
%     bench.append( lens32 );
%     bench.append( lens33 );
%     %----------------------------------

%     %double achromat with 6mm and 9mm FL
%         pos3= 0;
%         %45090 9mm FL
%     lens21 = Lens([pos3+0 0 0], 3,5.26,0,{'air' 'n-bk7'});
%     lens22 = Lens([pos3+1.47 0 0], 3,-3.98,0,{'n-bk7' 'sf5'});
%     lens23 = Lens([pos3+2.5 0 0], 3,-12.05,0,{'sf5' 'air'}); 
%     
%     %45089 6mm FL
%     pos3 = pos3+3.01;
%     lens31 = Lens([pos3+0 0 0], 3,4.13,0,{'air' 'baf10'});
%     lens32 = Lens([pos3+1.97 0 0], 3,-2.36,0,{'baf10' 'sf10'});
%     lens33 = Lens([pos3+3 0 0], 3,-21.70,0,{'sf10' 'air'}); 
%     
%     bench.append( lens21 );
%     bench.append( lens22 );
%     bench.append( lens23 );
%     bench.append( lens31 );
%     bench.append( lens32 );
%     bench.append( lens33 );
%     %----------------------------------
    
    nrays = 100;
    focal_point = [];
    for i=0:.01:.5
    %r = Rays( n, geometry, r, dir, D, pattern, glass, wavelength, color, diopter ) - object constructor
%         rays_in = Rays( nrays, 'collimated', [ -0.5 0 0 ], [ 1 i 0 ], .5, 'hexagonal', 'air',525*10^(-9),[ 0 1 0 ],1);
        rays_in = Rays( nrays, 'source', [ -1.42 i 0 ], [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9),[ 0 1 0 ],1);

        screen = Screen( [ 16  0 0 ], 10, 10, 256, 256 );
        bench.append( screen );

        rays_through = bench.trace( rays_in );    % repeat to get the min spread rays

        [f ff] = rays_through(end).focal_point();
        
        focal_point(end+1,:) = f([1 2]);
    end
    % draw bench elements and draw rays as arrows
%     figure(2)
    bench.draw( rays_through,'lines' );  % display everything, the other draw option is 'lines'
    daspect([1 1 1])
    view([0 0 1])
%     figure(3)
%     imshow( screen.image, [] );
%     figure(3)
%     plot(focal_point(:,2),focal_point(:,1),'.k')
end

