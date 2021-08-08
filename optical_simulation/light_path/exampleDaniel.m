function exampleDaniel()
    % create a container for optical elements (Bench class)
    bench = Bench;

    % add optical elements in the order they are encountered by light rays

    % aspheric 65-567, ~2.7mm FL
    pos1 = 1.9;
    lens11 = Plane( [ pos1 0 0 ], 4, { 'air' 'D-ZLaF52LA' } );
    %AsphericLens( r, D, R, k, avec, glass )
    a = [0 ...
        2.283443987985E-3 ...
        1.732258175861E-5 ...
        -1.277960734687E-5 ...
        -3.711789658434E-6];
    lens12 = AsphericLens( [pos1+1.43 0 0 ], 4, -1/4.623091887902032700E-001, -6.979185555E-1, a, { 'D-ZLaF52LA' 'air' } );
    
    bench.append( lens11 );
    bench.append( lens12 );
    %---------------------------------
    %Electrowetting lens
    pos2 = pos1+1.43+1;
    f=1000/0.000001;
    
    lens21 = Plane( [ pos2-1.7/2 0 0 ], 2.6, { 'air' 'bk7' } );
    % l = Lens( r, D, R, k, glass )
    n = 1.523; %N-BK7
    r = (n-1)*f;
    lens22 = Lens([pos2+1.7/2 0 0], 2.6,-1*r,0,{'bk7' 'air'});

    bench.append( lens21 );
    bench.append( lens22 );
    %----------------------------------
    %tube lens achromat 63691 FL = 10mm
    pos3= pos2+1.7/2+.2;
    lens31 = Lens([pos3+0 0 0], 4,7.12,0,{'air' 'baf10'});
    lens32 = Lens([pos3+2 0 0], 4,-4.22,0,{'baf10' 'sf10'});
    lens33 = Lens([pos3+3 0 0], 4,-33.66,0,{'sf10' 'air'});
    bench.append( lens31 );
    bench.append( lens32 );
    bench.append( lens33 );
    %----------------------------------
    
    % screen
    screen = Screen( [ pos3+3+8.6  0 0 ], 4, 4, 1024, 1024 );
    bench.append( screen );

    % create collimated rays with some slant
    numPS = 5;
    nrays = 100;
    colors = jet(numPS);
    xShift = zeros(numPS,1);%linspace(0,sqrt(.03),numPS).^2;
    yShift = linspace(0,.5,numPS);
    for i=1:numPS
        
        %r = Rays( n, geometry, r, dir, D, pattern, glass, wavelength, color, diopter ) - object constructor
        if i==1
            rays_in = Rays( nrays, 'source', [ 0 0 0 ], [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9),colors(i,:),1);
        else
            ray = Rays( nrays, 'source', [ xShift(i) yShift(i) 0 ], [ 1 0 0 ], .5, 'hexagonal', 'air',525*10^(-9),colors(i,:),1);
        	rays_in = rays_in.append(ray); 
        end
    end
    
    fprintf( 'Tracing rays... ' );
    rays_through = bench.trace( rays_in );    % repeat to get the min spread rays

    % draw bench elements and draw rays as arrows
%     figure(1)
    bench.draw( rays_through,'lines' );  % display everything, the other draw option is 'lines'
    view([0 0 1]);
    daspect([5 1 1]);
%     figure(2)
%     imshow( screen.image, [] );
%     axis([480 1000 480 520]);
%     daspect([1 .1 1])
end