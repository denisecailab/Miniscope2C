function lens = lensSpecs(partNumber, flipLens)
%LENSSPECS Summary of this function goes here
%   Detailed explanation goes here
  
    switch (partNumber)
        case 'achr_63691' %10mm FL
            lens.D = 4;
            lens.CT = [2 1];
            lens.R  = [7.12 -4.22 -33.66];
            lens.n  = {'baf10' 'sf10'};
            lens.EFL = 10;
            lens.BFL = 8.4;
        case 'achr_45090' %9mm FL
            lens.D = 3;
            lens.CT = [1.47 1.03];
            lens.R  = [5.26 -3.98 -12.05];
            lens.n  = {'n-bk7' 'sf5'};
            lens.EFL = 9;
            lens.BFL = 7.79;
        case 'achr_45089' %6mm FL
            lens.D = 3;
            lens.CT = [1.97 1.03];
            lens.R  = [4.13 -2.36 -21.70];
            lens.n  = {'baf10' 'sf10'};
            lens.EFL = 6;
            lens.BFL = 4.33;
        case 'achr_49924' %7.5mm FL, 6.25mm OD
            lens.D = 6.250;
            lens.CT = [5.00 1.00];
            lens.R  = [5.63 -3.71 -11.34];
            lens.n  = {'baf10' 'sf57'};
            lens.EFL = 7.5;
            lens.BFL = 4.45;
        case 'achr_45208' %10mm FL, 6.25mm OD
            lens.D = 6.250;
            lens.CT = [3.06 1.03];
            lens.R  = [6.98 -4.35 -41.01];
            lens.n  = {'baf10' 'sf10'};
            lens.EFL = 10;
            lens.BFL = 7.83;
        case 'achr_63694' %20mm FL, 10mm OD
            lens.D = 10.000;
            lens.CT = [4.00 1.00];
            lens.R  = [14.15 -8.38 -71.22];
            lens.n  = {'baf10' 'sf10'};
            lens.EFL = 20;
            lens.BFL = 17.25;
        case 'achr_65551' %15mm FL, 15mm OD
            lens.D = 15.000;
            lens.CT = [9.86 3.00];
            lens.R  = [11.39 -8.52 -24.69];
            lens.n  = {'basf64' 'sf66'};
            lens.EFL = 15;
            lens.BFL = 8.57;
        case 'achr_49929' %20mm FL, 15mm OD
            lens.D = 15.000;
            lens.CT = [6.80 2.00];
            lens.R  = [14.01 -11.36 -44.21];
            lens.n  = {'baf10' 'sf57'};
            lens.EFL = 20;
            lens.BFL = 15.34;
        case 'achr_65549' %20mm FL, 15mm OD
            lens.D = 9.000;
            lens.CT = [6.35 1.80];
            lens.R  = [6.87 -5 -14.16];
            lens.n  = {'basf64' 'sf66'};
            lens.EFL = 9;
            lens.BFL = 4.92;
        case 'achr_45206' %10mm FL, 5mm OD
            lens.D = 5.000;
            lens.CT = [1.73 1.03];
            lens.R = [7.17 -4.39 -33.96];
            lens.n = {'baf10' 'sf10'};
            lens.EFL = 10.00;
            lens.BFL = 8.56;
    end
    
    if flipLens==1
        lens.CT = flip(lens.CT);
        lens.R  = flip(lens.R)*-1;
        lens.n = flip(lens.n);
    end

end

