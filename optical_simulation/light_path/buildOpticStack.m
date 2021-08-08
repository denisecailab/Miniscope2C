function [bench xO xI  endLensPos] = buildOpticStack(bench, configuration,EWL_Power,startPosition)

    if (isempty(EWL_Power))
        EWL_Power = 0;
    end
    if isempty(startPosition)
        startPosition = 0;
    end
    pos = startPosition;
    
    switch (configuration)
        case 'Achr9_Achr6_EWL_Achr10'
                %% Achr 45-090, 9mm FL
                [lens1 CT ~] = buildAchrLens('achr_45090',1,pos); 
                bench.append(lens1{1});
                bench.append(lens1{2});
                bench.append(lens1{3});
                %% Achr 45-089, 6mm FL
                pos = pos + CT + 0.01;
                [lens2 CT ~] = buildAchrLens('achr_45089',1,pos); 
                bench.append(lens2{1});
                bench.append(lens2{2});
                bench.append(lens2{3});
                %% EWL 
                pos = pos + CT + 0.1;
                CT = 1.7;
                f=1000/EWL_Power;
                n = 1.523; %N-BK7
                r = (n-1)*f;
                lens3{1} = Plane([pos 0 0], 2.6, { 'air' 'bk7' } );  
                if EWL_Power == 0
                    lens3{2} = Plane([pos+1.7 0 0], 2.6, {'bk7' 'air'} ); 
                else
                    n = 1.523; %N-BK7
                    r = (n-1)*1000/EWL_Power;
                    lens3{2} = Lens([pos+1.7 0 0], 2.6,-1*r,0,{'bk7' 'air'});
                end
                bench.append(lens3{1});
                bench.append(lens3{2});
                %% Achr 63-691, 10mm FL
                pos = pos + CT + 0.1;
                
                [lens4 CT BFL] = buildAchrLens('achr_63691',0,pos); 
                bench.append(lens4{1});
                bench.append(lens4{2});
                bench.append(lens4{3});
                
                endLensPos = pos + CT;
                xO = -1.45+.025;
                xI = pos+CT + BFL;
            case 'Achr6_Achr6_EWL_Achr10'
                %% Achr 45-089, 6mm FL
                [lens1 CT ~] = buildAchrLens('achr_45089',1,pos); 
                bench.append(lens1{1});
                bench.append(lens1{2});
                bench.append(lens1{3});
                %% Achr 45-089, 6mm FL
                pos = pos + CT + 0.01;
                [lens2 CT ~] = buildAchrLens('achr_45089',1,pos); 
                bench.append(lens2{1});
                bench.append(lens2{2});
                bench.append(lens2{3});
                %% EWL 
                pos = pos + CT + 0.1;
                CT = 1.7;
                f=1000/EWL_Power;
                n = 1.523; %N-BK7
                r = (n-1)*f;
                lens3{1} = Plane([pos 0 0], 2.6, { 'air' 'bk7' } );  
                if EWL_Power == 0
                    lens3{2} = Plane([pos+1.7 0 0], 2.6, {'bk7' 'air'} ); 
                else
                    n = 1.523; %N-BK7
                    r = (n-1)*1000/EWL_Power;
                    lens3{2} = Lens([pos+1.7 0 0], 2.6,-1*r,0,{'bk7' 'air'});
                end
                bench.append(lens3{1});
                bench.append(lens3{2});
                %% Achr 63-691, 10mm FL
                pos = pos + CT + 0.1;
                
                [lens4 CT BFL] = buildAchrLens('achr_63691',0,pos); 
                bench.append(lens4{1});
                bench.append(lens4{2});
                bench.append(lens4{3});
                
                endLensPos = pos + CT;
                xI = pos+CT + BFL;
                %% Dichroic Mirror
                pos = pos + CT +BFL - 5;
                lens5{1} = Plane([pos 0 0], 6,4, { 'air' 'bk7' } ); 
                lens5{1}.rotate([0 0 1], -pi/4);
                lens5{2} = Plane([pos+sqrt(1/2) -1*sqrt(1/2) 0], 6,4, {  'bk7' 'air'} ); 
                lens5{2}.rotate([0 0 1], -pi/4);
                bench.append(lens5{1});
                bench.append(lens5{2});
                
                %% Emission Filter
                pos = pos + 4;
                lens5{1} = Plane([pos 0 0], 3,3, { 'air' 'bk7' } ); 
                lens5{2} = Plane([pos+1 0 0], 4,4, {  'bk7' 'air'} ); 
                bench.append(lens5{1});
                bench.append(lens5{2});
                
                xO = -0.734; % object location that sits on focal point
                xI = 20.3091;% when dichroic and filter are included
                
                
            case 'fret'
                %% Achr 45-089, 6mm FL
                [lens1 CT ~] = buildAchrLens('achr_45089',1,pos); 
                bench.append(lens1{1});
                bench.append(lens1{2});
                bench.append(lens1{3});
                %% Achr 45-089, 6mm FL
                pos = pos + CT + 0.01;
                [lens2 CT ~] = buildAchrLens('achr_45089',1,pos); 
                bench.append(lens2{1});
                bench.append(lens2{2});
                bench.append(lens2{3});
                %% EWL 
                pos = pos + CT + 0.1;
                CT = 1.7;
                f=1000/EWL_Power;
                n = 1.523; %N-BK7
                r = (n-1)*f;
                lens3{1} = Plane([pos 0 0], 2.6, { 'air' 'bk7' } );  
                if EWL_Power == 0
                    lens3{2} = Plane([pos+1.7 0 0], 2.6, {'bk7' 'air'} ); 
                else
                    n = 1.523; %N-BK7
                    r = (n-1)*1000/EWL_Power;
                    lens3{2} = Lens([pos+1.7 0 0], 2.6,-1*r,0,{'bk7' 'air'});
                end
                bench.append(lens3{1});
                bench.append(lens3{2});
                
                xO = -0.734; % object location that sits on focal point
                xI = 20.3091;% when dichroic and filter are included

    end

end