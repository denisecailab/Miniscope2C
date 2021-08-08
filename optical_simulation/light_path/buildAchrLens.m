function [surf, CT, BFL] = buildAchrLens( partNumber, flipLens, startPosition )
%BUILDACHRLENS Summary of this function goes here
%   Detailed explanation goes here
    pos = startPosition;
    lens = lensSpecs(partNumber,flipLens);
    surf{1} = Lens([pos 0 0], lens.D,lens.R(1),0,{'air' lens.n{1}});
    surf{2} = Lens([pos+lens.CT(1) 0 0], lens.D,lens.R(2),0,{lens.n{1} lens.n{2}});
    surf{3} = Lens([pos+sum(lens.CT) 0 0], lens.D,lens.R(3),0,{lens.n{2} 'air'});
    CT = sum(lens.CT);
    BFL = lens.BFL;
end

