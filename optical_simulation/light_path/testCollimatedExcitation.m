function screen = testCollimatedExcitation(EWL_Power)
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

%% Achr 45-089, 6mm FL
pos = 2;
% lens = lensSpecs('achr_45089',0);
% surf{1} = Lens([14.4 pos 0], lens.D,lens.R(1),0,{'air' lens.n{1}});
% surf{2} = Lens([14.4 pos+lens.CT(1) 0], lens.D,lens.R(2),0,{lens.n{1} lens.n{2}});
% surf{3} = Lens([14.4 pos+sum(lens.CT) 0], lens.D,lens.R(3),0,{lens.n{2} 'air'});

surf{1} = Lens([14.4 pos 0], 3,1.501,0,{'air' 'bk7'});
surf{2} = Plane([14.4 pos+1.5 0], 3,3,{'air' 'bk7'});
% surf{2} = Lens([14.4 pos+3 0], 3,1.51,0,{'air' 'bk7'});
surf{1}.rotate([0 0 1],pi/2);
surf{2}.rotate([0 0 1],-pi/2);
% surf{3}.rotate([0 0 1],pi/2);
bench.prepend(surf{1});
bench.prepend(surf{2});
% bench.prepend(surf{3});
%
%------------------------------------------------
%Add excitation filter
plane1 = Plane([14.4 pos+1.6 0], 4,4, {  'bk7' 'air'});
plane1.rotate([0 0 1],-pi/2);
plane2 = Plane([14.4 pos+2.6 0], 3,3, { 'air' 'bk7' });
plane2.rotate([0 0 1],-pi/2);
bench.prepend(plane1);
bench.prepend(plane2);
%------------------------------------------------

screen = Screen( [ -.7  0 0 ], 1, 1, 20, 20 );
    screen.rotate([0 0 1], pi);
    bench.append( screen );
    
% figure(1)
nrays = 300;
xShift = 0;
zShift = 0;
LEDPos = 6.5;%5.35
spread = .25;
rays_in = Rays( nrays, 'source', [ 14.4+xShift LEDPos 0], [ 0 -1 0 ], spread, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);

for xShift = -.3:.6:.3
    
    for zShift = -.3:.6:.3
        if ((xShift ~=0) || (zShift ~= 0))
            rays = Rays( nrays, 'source', [ 14.4+xShift LEDPos zShift], [ 0 -1 0 ], spread, 'hexagonal', 'air',480*10^(-9),[0 0 1],1);
            rays_in = rays_in.append(rays);
        end
    end
end
rays_through = bench.trace( rays_in );
bench.draw( rays_through, 'lines' );
view([0 0 1]);

figure(5)
s = screen;
pcolor(s.image)
colormap jet
daspect([1 1 1])
colorbar
% figure(6)
% a = screen.image;
% hist(a(:),100);
end