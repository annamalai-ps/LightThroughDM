#include <loop_device.hxx>

#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>
#include <limits>

namespace LightThroughDM {

constexpr int dim = 3;

// u(t,x,y,z) =
//   A cos(2 pi omega t) sin(2 pi kx x) sin(2 pi ky y) sin(2 pi kz z)
template <typename T>
constexpr void standing_wave(const T A, const T kx, const T ky, const T kz,
                             const T t, const T x, const T y, const T z,
                             T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));

  phi = A * cos(2 * pi * omega * t) * cos(2 * pi * kx * x) *
      cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  mu = A * (-2 * pi * omega) * sin(2 * pi * omega * t) * cos(2 * pi * kx * x) *
        cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  Ax = A * cos(2 * pi * omega * t) * cos(2 * pi * kx * x) *
      cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  nu = A * (-2 * pi * omega) * sin(2 * pi * omega * t) * cos(2 * pi * kx * x) *
        cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  Ay = A * cos(2 * pi * omega * t) * cos(2 * pi * kx * x) *
      cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  chi = A * (-2 * pi * omega) * sin(2 * pi * omega * t) * cos(2 * pi * kx * x) *
        cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  Az = A * cos(2 * pi * omega * t) * cos(2 * pi * kx * x) *
      cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
  psi = A * (-2 * pi * omega) * sin(2 * pi * omega * t) * cos(2 * pi * kx * x) *
        cos(2 * pi * ky * y) * cos(2 * pi * kz * z);
}

// u(t,r) = (f(t-r) - f(t+r)) / r
// f(v) = A exp(-1/2 (r/W)^2)
template <typename T>
constexpr void gaussian(const T A, const T W, const T x_offset, const T t, const T x, const T y,
                        const T z, T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi) {
  using std::exp, std::pow, std::sqrt;

  phi = A*exp(-( pow(x + x_offset,2.0) + pow(y,2.0) + pow(z,2.0)  )/(2.0*pow(W, 2)) );
  Ax = A*exp(-( pow(x + x_offset,2.0) + pow(y,2.0) + pow(z,2.0)  )/(2.0*pow(W, 2)) );
  Ay = A*exp(-( pow(x + x_offset,2.0) + pow(y,2.0) + pow(z,2.0)  )/(2.0*pow(W, 2)) );
  Az = A*exp(-( pow(x + x_offset,2.0) + pow(y,2.0) + pow(z,2.0)  )/(2.0*pow(W, 2)) );

  mu = 0.0;
  nu = 0.0;
  chi = 0.0;
  psi = 0.0;
  
}

extern "C" void LightThroughDM_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_Initial;
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(initial_condition, "standing wave")) {
    
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          standing_wave(amplitude, standing_wave_kx, standing_wave_ky,
                        standing_wave_kz, cctk_time, p.x, p.y, p.z, phi(p.I), mu(p.I),
                    Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I));
        });

  } else if (CCTK_EQUALS(initial_condition, "Gaussian")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          gaussian(amplitude, gaussian_width, gaussian_x_offset, cctk_time, p.x, p.y, p.z,  phi(p.I), mu(p.I),
                    Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I));
        });

  } else {
    CCTK_ERROR("Unknown initial condition");
  }
}

extern "C" void LightThroughDM_RHS(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_RHS;
  DECLARE_CCTK_PARAMETERS;

  if (CCTK_EQUALS(boundary_condition, "CarpetX")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          using std::pow;
          CCTK_REAL ddu = 0;
          for (int d = 0; d < dim; ++d)
            ddu += (u(p.I - p.DI[d]) - 2 * u(p.I) + u(p.I + p.DI[d])) /
                   pow(p.DX[d], 2);

          u_rhs(p.I) = rho(p.I);
          rho_rhs(p.I) = ddu;
        });

  } else if (CCTK_EQUALS(boundary_condition, "reflecting")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          using std::pow, std::sqrt;

          Arith::vect<CCTK_REAL, dim> dd_phi;
          Arith::vect<CCTK_REAL, dim> dd_Ax;
          Arith::vect<CCTK_REAL, dim> dd_Ay;
          Arith::vect<CCTK_REAL, dim> dd_Az;
          Arith::vect<CCTK_REAL, dim> d_phi;
          Arith::vect<CCTK_REAL, dim> d_Ax;
          Arith::vect<CCTK_REAL, dim> d_Ay;
          Arith::vect<CCTK_REAL, dim> d_Az;
          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0); 

          for (int d = 0; d < dim; ++d)
          {
            if (p.BI[d] < 0 || p.BI[d] > 0 ) //left and right boundaries
            {
              d_phi[d] = 0.0, d_Ax[d] = 0.0, d_Ay[d] = 0.0, d_Az[d] = 0.0;
              dd_phi[d] = 0.0, dd_Ax[d] = 0.0, dd_Ay[d] = 0.0, dd_Az[d] = 0.0;
            }
            else
            {
              d_phi[d] = (-phi(p.I + 2*p.DI[d]) + 8.0*phi(p.I + p.DI[d]) -8.0*phi(p.I - p.DI[d])
                       - phi(p.I - 2*p.DI[d])  )/(12.0*pow(p.DX[d], 2.0));
              d_Ax[d] = (-Ax(p.I + 2*p.DI[d]) + 8.0*Ax(p.I + p.DI[d]) -8.0*Ax(p.I - p.DI[d])
                       - Ax(p.I - 2*p.DI[d])  )/(12.0*pow(p.DX[d], 2.0));
              d_Ay[d] = (-Ay(p.I + 2*p.DI[d]) + 8.0*Ay(p.I + p.DI[d]) -8.0*Ay(p.I - p.DI[d])
                       - Ay(p.I - 2*p.DI[d])  )/(12.0*pow(p.DX[d], 2.0));
              d_Az[d] = (-Az(p.I + 2*p.DI[d]) + 8.0*Az(p.I + p.DI[d]) -8.0*Az(p.I - p.DI[d])
                       - Az(p.I - 2*p.DI[d])  )/(12.0*pow(p.DX[d], 2.0));

              dd_phi[d] = ( -phi(p.I + 2*p.DI[d]) + 16.0*phi(p.I + p.DI[d]) - 30.0*phi(p.I) + 
                        16.0*phi(p.I - p.DI[d]) - phi(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
              dd_Ax[d] = ( -Ax(p.I + 2*p.DI[d]) + 16.0*Ax(p.I + p.DI[d]) - 30.0*Ax(p.I) + 
                        16.0*Ax(p.I - p.DI[d]) - Ax(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
              dd_Ay[d] = ( -Ay(p.I + 2*p.DI[d]) + 16.0*Ay(p.I + p.DI[d]) - 30.0*Ay(p.I) + 
                        16.0*Ay(p.I - p.DI[d]) - Ay(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
              dd_Az[d] = ( -Az(p.I + 2*p.DI[d]) + 16.0*Az(p.I + p.DI[d]) - 30.0*Az(p.I) + 
                        16.0*Az(p.I - p.DI[d]) - Az(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );

            }
          }
          if (sqrt(r_square) >= a_ext) // exterior
          {
            const CCTK_REAL alpha_ext = M / sqrt(r_square);
            Arith::vect<CCTK_REAL, dim> d_alpha_ext;
            Arith::vect<CCTK_REAL, dim> dd_alpha_ext;

            for (int d = 0; d < dim; ++d) {
              d_alpha_ext[d] = -(M*p.X[d]) / pow(r_square,1.5);
              dd_alpha_ext[d] = M*((3.0*pow(p.X[d], 2.0)) / pow(r_square,2.5) - 1.0/pow(r_square,1.5)); 
            }

            CCTK_REAL rho_ext = 0.0;
            CCTK_REAL P_ext = 0.0;

            phi_rhs(p.I) = mu(p.I);
            mu_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                          -phi(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*nu(p.I) + d_alpha_ext[1]*chi(p.I) + d_alpha_ext[2]*psi(p.I))
                          - 2.0*(d_alpha_ext[0]*d_phi[0] + d_alpha_ext[1]*d_phi[1] + d_alpha_ext[2]*d_phi[2])
                          + 4.0*M_PI*(rho_ext + 3.0*P_ext)*phi(p.I) );
            Ax_rhs(p.I) = nu(p.I);
            nu_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                          +Ax(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[2]*d_Az[0] + d_alpha_ext[1]*d_Ay[0])
                          - 2.0*d_alpha_ext[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[0] + d_alpha_ext[1]*d_Ax[1] + d_alpha_ext[2]*d_Ax[2])
                          - 4.0*M_PI*(rho_ext - P_ext)*Ax(p.I) );
            Ay_rhs(p.I) = chi(p.I);
            chi_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                          +Ay(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[1] + d_alpha_ext[2]*d_Az[1])
                          - 2.0*d_alpha_ext[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                          + 2.0*(d_alpha_ext[0]*d_Ay[0] + d_alpha_ext[1]*d_Ay[1] + d_alpha_ext[2]*d_Ay[2])
                          - 4.0*M_PI*(rho_ext - P_ext)*Ay(p.I) );  
            Ax_rhs(p.I) = psi(p.I);
            psi_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                          +Az(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[2] + d_alpha_ext[1]*d_Ay[2])
                          - 2.0*d_alpha_ext[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                          + 2.0*(d_alpha_ext[0]*d_Az[0] + d_alpha_ext[1]*d_Az[1] + d_alpha_ext[2]*d_Az[2])
                          - 4.0*M_PI*(rho_ext - P_ext)*Az(p.I) );
          }
          else //interior
          {

            const CCTK_REAL alpha_int = ( M / 2.0 * a_ext )*(3.0 - r_square/pow(a_ext, 2.0) );
            const CCTK_REAL dd_alpha_int = M/pow(a_ext, 3.0);
            Arith::vect<CCTK_REAL, dim> d_alpha_int;
            for (int d = 0; d < dim; ++d)
              d_alpha_int[d] = -(M*p.X[d]) / pow(a_ext, 3.0);

            CCTK_REAL constC = sqrt(M/(2.0*pow(a_ext,3.0)));
            CCTK_REAL constA = constC*(4.0*a_ext - M)/(2.0*a_ext + M);
            CCTK_REAL constB = (1.0/constC)*(2.0*a_ext - 2.0*M)/(2.0*a_ext + M);
            CCTK_REAL rho_int = (3.0*M)/((4.0*M_PI*pow(a_ext,3.0))*pow(1 + M/(2.0*a_ext),6.0));
            CCTK_REAL R2 = 3.0/(8.0*M_PI*rho_int);
            CCTK_REAL P_int = ( constA*( pow(constC,-2.0) - 2.0*r_square ) 
                              + constB*(r_square*pow(constC,2.0) - 2.0) )/( 8.0*M_PI*R2*(constA*r_square + constB) );

            phi_rhs(p.I) = mu(p.I);
            mu_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                          -phi(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*nu(p.I) + d_alpha_int[1]*chi(p.I) + d_alpha_int[2]*psi(p.I))
                          - 2.0*(d_alpha_int[0]*d_phi[0] + d_alpha_int[1]*d_phi[1] + d_alpha_int[2]*d_phi[2])
                          + 4.0*M_PI*(rho_int + 3.0*P_int)*phi(p.I) );
            Ax_rhs(p.I) = nu(p.I);
            nu_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                          +Ax(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[2]*d_Az[0] + d_alpha_int[1]*d_Ay[0])
                          - 2.0*d_alpha_int[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                          + 2.0*(d_alpha_int[0]*d_Ax[0] + d_alpha_int[1]*d_Ax[1] + d_alpha_int[2]*d_Ax[2])
                          - 4.0*M_PI*(rho_int - P_int)*Ax(p.I) );
            Ay_rhs(p.I) = chi(p.I);
            chi_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                          +Ay(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*d_Ax[1] + d_alpha_int[2]*d_Az[1])
                          - 2.0*d_alpha_int[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                          + 2.0*(d_alpha_int[0]*d_Ay[0] + d_alpha_int[1]*d_Ay[1] + d_alpha_int[2]*d_Ay[2])
                          - 4.0*M_PI*(rho_int - P_int)*Ay(p.I) );  
            Ax_rhs(p.I) = psi(p.I);
            psi_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                          +Az(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*d_Ax[2] + d_alpha_int[1]*d_Ay[2])
                          - 2.0*d_alpha_int[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                          + 2.0*(d_alpha_int[0]*d_Az[0] + d_alpha_int[1]*d_Az[1] + d_alpha_int[2]*d_Az[2])
                          - 4.0*M_PI*(rho_int - P_int)*Az(p.I) );

          }
        });
  
  } else {
    CCTK_ERROR("Specify proper boundary condition");
  }
}

} // namespace LightThroughDM
