%% Setting the method and geometry
Computation = 'Ground';
Ncomponents = 2;
Type = 'BESP';
Deltat = 1e-2;
Stop_time = [];
Stop_crit = {'Energy',1e-10};
Method = Method_Var3d(Computation, Ncomponents, Type, Deltat, Stop_time, Stop_crit);
xmin = -10;
xmax = 10;
ymin = -10;
ymax = 10;
zmin = -10;
zmax = 10;
Nx = 2^6+1;
Ny = 2^6+1;
Nz = 2^6+1;
Geometry3D = Geometry3D_Var3d(xmin,xmax,ymin,ymax,zmin,zmax,Nx,Ny,Nz);

%% 

% function [CoupledCubicNonlinearity] = Coupled_Cubic2d(Beta_coupled)
%     CoupledCubicNonlinearity = cell(2);
%     CoupledCubicNonlinearity{1,1} = @(Phi,X,Y) Beta_coupled(1,1)*abs(Phi{1}).^2+...
%     Beta_coupled(1,2)*abs(Phi{2}).^2;
%     CoupledCubicNonlinearity{2,2} = @(Phi,X,Y) Beta_coupled(2,2)*abs(Phi{2}).^2+...
%     Beta_coupled(2,1)*abs(Phi{1}).^2;
%     CoupledCubicNonlinearity{1,2} = @(Phi,X,Y) 0;
%     CoupledCubicNonlinearity{2,1} = @(Phi,X,Y) 0;
% end
% 
% function [CoupledCubicEnergy] = Coupled_Cubic_energy2d(Method,Beta_coupled)
%     CoupledCubicEnergy = cell(2);
%     CoupledCubicEnergy{1,1} = @(Phi,X,Y) (1/2)*Beta_coupled(1,1)*abs(Phi{1}).^2+...
%     (1/2)*Beta_coupled(1,2)*abs(Phi{2}).^2;
%     CoupledCubicEnergy{2,2} = @(Phi,X,Y) (1/2)*Beta_coupled(2,2)*abs(Phi{2}).^2+...
%     (1/2)*Beta_coupled(2,1)*abs(Phi{1}).^2;
%     CoupledCubicEnergy{1,2} = @(Phi,X,Y) 0;
%     CoupledCubicEnergy{2,1} = @(Phi,X,Y) 0;
% end
% 
% function [Dipolar_interaction_nonlinearity]= Dipolar_interaction3d(Phi, FFTX,...
%     FFTY, FFTZ, Dipolar_direction, d)
%     Cross_norm = sqrt((FFTY*Dipolar_direction(3)-FFTZ*Dipolar_direction(2)).^2+ ...
%         +(FFTZ*Dipolar_direction(1)-FFTX*Dipolar_direction(3)).^2 ...
%         +(FFTX*Dipolar_direction(2)-FFTY*Dipolar_direction(1)).^2);
%     Scalar_prod = FFTX*Dipolar_direction(1)+FFTY*Dipolar_direction(2)...
%         +FFTZ*Dipolar_direction(3);
%     Angle = atan2(Cross_norm,Scalar_prod);
%     NLFFT = fftn(abs(Phi).^2);
%     V = d^2*(4/3)*pi*(3*cos(Angle).^2-1);
%     Dipolar_interaction_nonlinearity = ifftn(V.*NLFFT);
% end
% 
% Delta = 0.5;
% Beta = 200;
% Beta_coupled= [2,1;1,1];
% Omega = 0.8;
% Physics2D = Physics2D_Var2d(Method,Delta,[],Omega);
% Physics2D = Potential_Var2d(Method, Physics2D);
% Physics2D = Nonlinearity_Var2d(Method, Physics2D,...
% Coupled_Cubic2d(Method,Beta_coupled),...
% [],Coupled_Cubic_energy2d(Method,Beta_coupled));
% Physics2D = Gradientx_Var2d(Method, Physics2D);
% Physics2D = Gradienty_Var2d(Method, Physics2D);
% 
% Delta = 0.5;
% Beta = 1000;
% Dipolar_direction =[0,0,1];
% d = 0.5;
% Physics3D = Physics3D_Var3d(Method,Delta,Beta);
% Physics3D = Potential_Var3d(Method, Physics3D);
% Physics3D = Nonlinearity_Var3d(Method, Physics2D);
% Physics3D = FFTNonlinearity_Var3d(Method, Physics3D,...
%     @(Phi,X,Y,Z,FFTX,FFTY,FFTZ)Dipolar_interaction3d(Phi, FFTX, FFTY, FFTZ, ...
%     Dipolar_direction ,d));

%% 

function Fdd = SpinDipoleFFT3d(Kdd)
    % Kdd{alpha,beta,gamma,delta} is a function handle:
    % Kdd{a,b,g,d}(FFTX,FFTY,FFTZ)
    %
    % Components:
    % 1 = +
    % 2 = -
    
    Fdd = cell(2,2);
    
    for alpha = 1:2
        for delta = 1:2
            a = alpha;
            d = delta;
    
            Fdd{a,d} = @(Phi,X,Y,Z,FFTX,FFTY,FFTZ) ...
                SpinDipoleEntry3d(Phi,FFTX,FFTY,FFTZ,Kdd,a,d);
        end
    end
end

function out = SpinDipoleEntry3d(Phi,FFTX,FFTY,FFTZ,Kdd,alpha,delta)
    out = zeros(size(Phi{1}));
    
    for beta = 1:2
        for gamma = 1:2
    
            rho_bg = conj(Phi{beta}).*Phi{gamma};
    
            K = Kdd{alpha,beta,gamma,delta}(FFTX,FFTY,FFTZ);
    
            out = out + ifftn(K .* fftn(rho_bg));
    
        end
    end
end

function K = DipoleKernel3d(FFTX,FFTY,FFTZ,e,C)

    k2 = FFTX.^2 + FFTY.^2 + FFTZ.^2;
    kdot = FFTX*e(1) + FFTY*e(2) + FFTZ*e(3);
    
    K = C*(4*pi/3)*(3*(kdot.^2)./(k2 + (k2==0)) - 1);
    
    K(k2==0) = 0;

end

e = [0,0,1];

C = zeros(2,2,2,2);

Cpppp = 1.0;  % ++ -> ++
Cmmmm = 1.0;  % -- -> --
Cpmmp = 0.5;
Cmppm = 0.5;
Cppmm = 0.2; % -- -> ++ type channel
Cmmpp = 0.2; % ++ -> -- type channel

% Examples:
C(1,1,1,1) = Cpppp;   % ++ -> ++
C(2,2,2,2) = Cmmmm;   % -- -> --
C(1,2,2,1) = Cpmmp;  % example density-like +- channel
C(2,1,1,2) = Cmppm;

% Spin-exchange / coherent dipolar channels, if allowed:
C(1,1,2,2) = Cppmm;   % -- -> ++ type channel
C(2,2,1,1) = Cmmpp;   % ++ -> -- type channel

Kdd = cell(2,2,2,2);

for alpha = 1:2
    for beta = 1:2
        for gamma = 1:2
            for delta = 1:2

                Cabgd = C(alpha,beta,gamma,delta);

                Kdd{alpha,beta,gamma,delta} = ...
                    @(FFTX,FFTY,FFTZ) DipoleKernel3d(FFTX,FFTY,FFTZ,e,Cabgd);

            end
        end
    end
end

Fdd = SpinDipoleFFT3d(Kdd);

Physics3D = Physics3D_Var3d(Method,Delta,Beta);
Physics3D = Potential_Var3d(Method, Physics3D);
Physics3D = FFTNonlinearity_Var3d(Method,Physics3D,Fdd);

%% Setting the initial data
InitialData_Choice = 2;
Phi_0 = InitialData_Var3d(Method, Geometry3D, Physics3D, InitialData_Choice);

%% Setting informations and outputs
Outputs = OutputsINI_Var3d(Method);
Printing = 1;
Evo = 15;
Draw = 1;
Print = Print_Var2d(Printing,Evo,Draw);

%-----------------------------------------------------------
% Launching simulation
%-----------------------------------------------------------

[Phi, Outputs] = GPELab3d(Phi_0,Method,Geometry3D,Physics3D,Outputs,[],Print);