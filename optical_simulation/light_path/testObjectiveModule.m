function testObjectiveModule( EWL_Power )
nrays = 100;

rays_Emission = Rays( nrays, 'collimated', [1 0 0], [ 1 0 0 ], 2, 'hexagonal', 'air',525*10^(-9),[ 0 1 0],1);
benchTemp = Bench;
bench = Bench;
[benchTemp xO xI] = buildOpticStack(benchTemp,'Achr6_Achr6_EWL_Achr10',EWL_Power,0);

screen = Screen( [26 0 0 ], 3, 3, 1000, 1000 );
% screen.rotate([0 0 1],pi);

bench.append(benchTemp.elem(9:end));
bench.cnt = length(bench.elem);
bench.append( screen );
% bench.elem = flip(bench.elem);
% bench.append(benchTemp.elem(2))
% bench.append(benchTemp.elem(3))


rays_through = bench.trace( rays_Emission );
% rays_through(end).
rays_through( end ).focal_point
bench.draw( rays_through, 'lines' );
% view([0 0 1]);



end

