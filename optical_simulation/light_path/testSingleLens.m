function  testSingleLens()
%TESTSINGLELENS Summary of this function goes here
%   Detailed explanation goes here

    bench = Bench;
    pos = 0;
%     %% Achr 45-089, 6mm FL
%     [lens1 CT ~] = buildAchrLens('achr_45089',0,pos);
%     bench.append(lens1{1});
%     bench.append(lens1{2});
%     bench.append(lens1{3});
%     
%     %% Achr 45-089, 6mm FL
%     pos = pos+CT+0.1;
%     [lens1 CT ~] = buildAchrLens('achr_45089',0,pos);
%     bench.append(lens1{1});
%     bench.append(lens1{2});
%     bench.append(lens1{3});
    %% achr_63691, 10mm FL
    [lens1 CT BFL] = buildAchrLens('achr_63691',0,pos);
    bench.append(lens1{1});
    bench.append(lens1{2});
    bench.append(lens1{3});
    
    %% Dichroic Mirror
%     pos = pos + CT +BFL - 5;
%     lens5{1} = Plane([pos 0 0], 6,4, { 'air' 'bk7' } ); 
%     lens5{1}.rotate([0 0 1], -pi/4);
%     lens5{2} = Plane([pos+sqrt(1/2) -1*sqrt(1/2) 0], 6,4, {  'bk7' 'air'} ); 
%     lens5{2}.rotate([0 0 1], -pi/4);
%     bench.append(lens5{1});
%     bench.append(lens5{2});
    %%
    screen = Screen( [ 14  0 0 ], 4, 4, 1024, 1024 );
    bench.append( screen );
    yAng = linspace(-.15,.15, 31);
    f = [];
    ff = [];
    for i = yAng
        nrays = 1000;
        rays_in = Rays( nrays, 'collimated', [ -1 0 0], [ 1 i 0 ], 2.5, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);
        rays_through = bench.trace( rays_in );

        [f(:,end+1) ff(end+1)] = rays_through(end).focal_point();
    end
%     f(1)-pos-CT
%     f
    plot(f(2,:)-f(2,16),f(1,:)-f(1,16),'b','linewidth',2)
%     bench.draw( rays_through,'lines',0.33,1,1);
%     view([0 0 1])
end

