#include <loop_device.hxx>

#include <vect.hxx>

#include <cctk.h>
#include <cctk_Arguments.h>
#include <cctk_Parameters.h>

#include <cmath>
#include <limits>

namespace LightThroughDM {

constexpr int dim = 3;



template <typename T>
constexpr void plane_wave(const T M, const T sigma, const T A, const T kx, const T ky, const T kz,
                             const T t, const T x, const T y, const T z, T &density,
                             T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi,
                             T &phi_flat, T &mu_flat, T &Ax_flat, T &nu_flat, T &Ay_flat, T &chi_flat, T &Az_flat, T &psi_flat) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt, std::erf, std::exp;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));
  const T r_inv_cubed = pow((pow(x,2.0)+pow(y,2.0)+pow(z,2.0)),-1.5);
  const T r_square = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);

  density = ( M*pow(sigma,-3.0)*pow(2.0*pi,-1.5) )*exp(-r_square/(2.0*pow(sigma,2.0)));

  const T n1 = 7;
  const T n2 = 11; // wave numbers between two full waves

  if (z >= (n1/(4.0*omega)) && z <= (n2/(4.0*omega)) )
  {
    Ax = A*cos(2*pi*omega*(z + t));
    nu = -2.0*A*pi*omega*sin(2*pi*omega*(z + t));
    Ay = A*sin(2*pi*omega*(z + t));
    chi = 2.0*A*pi*omega*cos(2*pi*omega*(z + t));
    Az = 0.0;
    psi = 0.0;
    phi = M*A*pow(pi*omega,-1.0)*( x*sin(2*pi*omega*(z + t)) - y*cos(2*pi*omega*(z + t)) )*( erf(sqrt(r_square)/(sqrt(2)*sigma))*r_inv_cubed
          - sqrt(2/pi)*exp(-r_square/(2*pow(sigma,2.0)))/(sigma*r_square) );
    mu = 2.0*M*A*( x*cos(2*pi*omega*(z + t)) + y*sin(2*pi*omega*(z + t))  )*( erf(sqrt(r_square)/(sqrt(2)*sigma))*r_inv_cubed
          - sqrt(2/pi)*exp(-r_square/(2*pow(sigma,2.0)))/(sigma*r_square) );

    Ax_flat = Ax;
    nu_flat = nu;
    Ay_flat = Ay;
    chi_flat = chi;
    Az_flat = Az;
    psi_flat = psi;
    phi_flat = 0.0;
    mu_flat = 0.0;
  }
  else
  {
    phi = 0.0, mu = 0.0, Ax = 0.0, nu = 0.0, Ay = 0.0, chi = 0.0, Az = 0.0, psi = 0.0;
    phi_flat = 0.0, mu_flat = 0.0, Ax_flat = 0.0, nu_flat = 0.0, Ay_flat = 0.0, chi_flat = 0.0, Az_flat = 0.0, psi_flat = 0.0;
  }

}

extern "C" void LightThroughDM_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_Initial;
  DECLARE_CCTK_PARAMETERS;
    
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          
          if (CCTK_EQUALS(initial_condition, "plane wave")) {
            plane_wave(M, sigma, wave_amplitude, plane_wave_kx, plane_wave_ky,
                          plane_wave_kz, cctk_time, p.x, p.y, p.z, density(p.I),
                          phi(p.I), mu(p.I), Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I),
                          phi_flat(p.I), mu_flat(p.I), Ax_flat(p.I), nu_flat(p.I), Ay_flat(p.I), chi_flat(p.I), Az_flat(p.I), psi_flat(p.I));
            
          }
          else {
            CCTK_ERROR("Unknown initial condition");
          }
        });  
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
            ddu += (phi(p.I - p.DI[d]) - 2 * phi(p.I) + phi(p.I + p.DI[d])) /
                   pow(p.DX[d], 2);

          phi_rhs(p.I) = mu(p.I);
          mu_rhs(p.I) = ddu;
        });

  } else if (CCTK_EQUALS(boundary_condition, "periodic")) {

    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          using std::acos, std::pow, std::sqrt, std::erf, std::exp;

          Arith::vect<CCTK_REAL, dim> dd_phi;
          Arith::vect<CCTK_REAL, dim> dd_Ax;
          Arith::vect<CCTK_REAL, dim> dd_Ay;
          Arith::vect<CCTK_REAL, dim> dd_Az;          
          Arith::vect<CCTK_REAL, dim> dd_phi_flat;
          Arith::vect<CCTK_REAL, dim> dd_Ax_flat;
          Arith::vect<CCTK_REAL, dim> dd_Ay_flat;
          Arith::vect<CCTK_REAL, dim> dd_Az_flat;
          Arith::vect<CCTK_REAL, dim> d_phi;
          Arith::vect<CCTK_REAL, dim> d_Ax;
          Arith::vect<CCTK_REAL, dim> d_Ay;
          Arith::vect<CCTK_REAL, dim> d_Az;

          const CCTK_REAL pi = acos(-1.0);
          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0);
          const CCTK_REAL r = sqrt(r_square);
          const CCTK_REAL r_inv_cubed = pow((pow(p.x,2.0)+pow(p.y,2.0)+pow(p.z,2.0)),-1.5);

          for (int d = 0; d < dim; ++d)
          {
            d_phi[d] = (-phi(p.I + 2*p.DI[d]) + 8.0*phi(p.I + p.DI[d]) -8.0*phi(p.I - p.DI[d])
                     + phi(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ax[d] = (-Ax(p.I + 2*p.DI[d]) + 8.0*Ax(p.I + p.DI[d]) -8.0*Ax(p.I - p.DI[d])
                     + Ax(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ay[d] = (-Ay(p.I + 2*p.DI[d]) + 8.0*Ay(p.I + p.DI[d]) -8.0*Ay(p.I - p.DI[d])
                     + Ay(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Az[d] = (-Az(p.I + 2*p.DI[d]) + 8.0*Az(p.I + p.DI[d]) -8.0*Az(p.I - p.DI[d])
                     + Az(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);

            dd_phi[d] = ( -phi(p.I + 2*p.DI[d]) + 16.0*phi(p.I + p.DI[d]) - 30.0*phi(p.I) + 
                      16.0*phi(p.I - p.DI[d]) - phi(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ax[d] = ( -Ax(p.I + 2*p.DI[d]) + 16.0*Ax(p.I + p.DI[d]) - 30.0*Ax(p.I) + 
                      16.0*Ax(p.I - p.DI[d]) - Ax(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ay[d] = ( -Ay(p.I + 2*p.DI[d]) + 16.0*Ay(p.I + p.DI[d]) - 30.0*Ay(p.I) + 
                      16.0*Ay(p.I - p.DI[d]) - Ay(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Az[d] = ( -Az(p.I + 2*p.DI[d]) + 16.0*Az(p.I + p.DI[d]) - 30.0*Az(p.I) + 
                      16.0*Az(p.I - p.DI[d]) - Az(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );            
            
            dd_phi_flat[d] = ( -phi_flat(p.I + 2*p.DI[d]) + 16.0*phi_flat(p.I + p.DI[d]) - 30.0*phi_flat(p.I) + 
                      16.0*phi_flat(p.I - p.DI[d]) - phi_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ax_flat[d] = ( -Ax_flat(p.I + 2*p.DI[d]) + 16.0*Ax_flat(p.I + p.DI[d]) - 30.0*Ax_flat(p.I) + 
                      16.0*Ax_flat(p.I - p.DI[d]) - Ax_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ay_flat[d] = ( -Ay_flat(p.I + 2*p.DI[d]) + 16.0*Ay_flat(p.I + p.DI[d]) - 30.0*Ay_flat(p.I) + 
                      16.0*Ay_flat(p.I - p.DI[d]) - Ay_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Az_flat[d] = ( -Az_flat(p.I + 2*p.DI[d]) + 16.0*Az_flat(p.I + p.DI[d]) - 30.0*Az_flat(p.I) + 
                      16.0*Az_flat(p.I - p.DI[d]) - Az_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
          
          }

          // flat spacetime Maxwell's eqns rhs
          phi_rhs_flat(p.I) = mu_flat(p.I);
          mu_rhs_flat(p.I) = (dd_phi_flat[0] + dd_phi_flat[1] + dd_phi_flat[2]);
          Ax_rhs_flat(p.I) = nu_flat(p.I);
          nu_rhs_flat(p.I) = (dd_Ax_flat[0] + dd_Ax_flat[1] + dd_Ax_flat[2]);
          Ay_rhs_flat(p.I) = chi_flat(p.I);
          chi_rhs_flat(p.I) = (dd_Ay_flat[0] + dd_Ay_flat[1] + dd_Ay_flat[2]);
          Az_rhs_flat(p.I) = psi_flat(p.I);
          psi_rhs_flat(p.I) = (dd_Az_flat[0] + dd_Az_flat[1] + dd_Az_flat[2]);

          const CCTK_REAL alpha = M*pow(r,-1.0)*erf(r/(sqrt(2.0)*sigma));
          Arith::vect<CCTK_REAL, dim> d_alpha;
          const CCTK_REAL sum_dd_alpha = -sqrt(2.0/pi)*(M*pow(sigma,-3.0))*exp(-r_square/(2.0*pow(sigma,2.0)));

          for (int d = 0; d < dim; ++d) {
            d_alpha[d] = (M*p.X[d])*( sqrt(2/pi)*exp(-r_square/(2.0*pow(sigma,2.0)))/(sigma*r_square) 
                          -erf(r/(sqrt(2.0)*sigma))*r_inv_cubed );
          }

          phi_rhs(p.I) = mu(p.I);
          mu_rhs(p.I) = pow(1 + 2.0*alpha,-1.0)*( (1 - 2.0*alpha)*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                        -phi(p.I)*sum_dd_alpha
                        + 2.0*(d_alpha[0]*nu(p.I) + d_alpha[1]*chi(p.I) + d_alpha[2]*psi(p.I))
                        - 2.0*(d_alpha[0]*d_phi[0] + d_alpha[1]*d_phi[1] + d_alpha[2]*d_phi[2])
                        + 4.0*M_PI*density(p.I)*phi(p.I) );
          Ax_rhs(p.I) = nu(p.I);
          nu_rhs(p.I) = pow(1 + 2.0*alpha,-1.0)*( (1 - 2.0*alpha)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                        +Ax(p.I)*sum_dd_alpha
                        + 2.0*(d_alpha[2]*d_Az[0] + d_alpha[1]*d_Ay[0])
                        - 2.0*d_alpha[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                        + 2.0*(d_alpha[0]*d_Ax[0] + d_alpha[1]*d_Ax[1] + d_alpha[2]*d_Ax[2])
                        - 4.0*M_PI*density(p.I)*Ax(p.I) );
          Ay_rhs(p.I) = chi(p.I);
          chi_rhs(p.I) = pow(1 + 2.0*alpha,-1.0)*( (1 - 2.0*alpha)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                        +Ay(p.I)*sum_dd_alpha
                        + 2.0*(d_alpha[0]*d_Ax[1] + d_alpha[2]*d_Az[1])
                        - 2.0*d_alpha[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                        + 2.0*(d_alpha[0]*d_Ay[0] + d_alpha[1]*d_Ay[1] + d_alpha[2]*d_Ay[2])
                        - 4.0*M_PI*density(p.I)*Ay(p.I) );  
          Az_rhs(p.I) = psi(p.I);
          psi_rhs(p.I) = pow(1 + 2.0*alpha,-1.0)*( (1 - 2.0*alpha)*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                        +Az(p.I)*sum_dd_alpha
                        + 2.0*(d_alpha[0]*d_Ax[2] + d_alpha[1]*d_Ay[2])
                        - 2.0*d_alpha[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                        + 2.0*(d_alpha[0]*d_Az[0] + d_alpha[1]*d_Az[1] + d_alpha[2]*d_Az[2])
                        - 4.0*M_PI*density(p.I)*Az(p.I) );

        });
  
  } else {
    CCTK_ERROR("Specify proper boundary condition");
  }
}

extern "C" void LightThroughDM_Constraint(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_Constraint;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          using std::acos, std::pow, std::sqrt, std::erf, std::exp;

          Arith::vect<CCTK_REAL, dim> d_phi;
          Arith::vect<CCTK_REAL, dim> d_Ax;
          Arith::vect<CCTK_REAL, dim> d_Ay;
          Arith::vect<CCTK_REAL, dim> d_Az;
          Arith::vect<CCTK_REAL, dim> d_phi_flat;
          Arith::vect<CCTK_REAL, dim> d_Ax_flat;
          Arith::vect<CCTK_REAL, dim> d_Ay_flat;
          Arith::vect<CCTK_REAL, dim> d_Az_flat;

          const CCTK_REAL pi = acos(-1.0);
          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0);
          const CCTK_REAL r = sqrt(r_square);
          const CCTK_REAL r_inv_cubed = pow((pow(p.x,2.0)+pow(p.y,2.0)+pow(p.z,2.0)),-1.5);


          for (int d = 0; d < dim; ++d)
          {
            d_phi[d] = (-phi(p.I + 2*p.DI[d]) + 8.0*phi(p.I + p.DI[d]) -8.0*phi(p.I - p.DI[d])
                     + phi(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ax[d] = (-Ax(p.I + 2*p.DI[d]) + 8.0*Ax(p.I + p.DI[d]) -8.0*Ax(p.I - p.DI[d])
                     + Ax(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ay[d] = (-Ay(p.I + 2*p.DI[d]) + 8.0*Ay(p.I + p.DI[d]) -8.0*Ay(p.I - p.DI[d])
                     + Ay(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Az[d] = (-Az(p.I + 2*p.DI[d]) + 8.0*Az(p.I + p.DI[d]) -8.0*Az(p.I - p.DI[d])
                     + Az(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);


            d_phi_flat[d] = (-phi_flat(p.I + 2*p.DI[d]) + 8.0*phi_flat(p.I + p.DI[d]) -8.0*phi_flat(p.I - p.DI[d])
                     + phi_flat(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ax_flat[d] = (-Ax_flat(p.I + 2*p.DI[d]) + 8.0*Ax_flat(p.I + p.DI[d]) -8.0*Ax_flat(p.I - p.DI[d])
                     + Ax_flat(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Ay_flat[d] = (-Ay_flat(p.I + 2*p.DI[d]) + 8.0*Ay_flat(p.I + p.DI[d]) -8.0*Ay_flat(p.I - p.DI[d])
                     + Ay_flat(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            d_Az_flat[d] = (-Az_flat(p.I + 2*p.DI[d]) + 8.0*Az_flat(p.I + p.DI[d]) -8.0*Az_flat(p.I - p.DI[d])
                     + Az_flat(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
          }

          // for flat spacetime
          constraint_violation_flat(p.I) = ( mu_flat(p.I) + d_Ax_flat[0] + d_Ay_flat[1] + d_Az_flat[2] );

          Arith::vect<CCTK_REAL, dim> d_alpha;
          for (int d = 0; d < dim; ++d) {
          d_alpha[d] = (M*p.X[d])*( sqrt(2/pi)*exp(-r_square/(2.0*pow(sigma,2.0)))/(sigma*r_square) 
                        -erf(r/(sqrt(2.0)*sigma))*r_inv_cubed );
          }
      
          constraint_violation(p.I) = ( mu(p.I) + d_Ax[0] + d_Ay[1] + d_Az[2] ) 
                                      + 2.0*(d_alpha[0]*Ax(p.I) + d_alpha[1]*Ay(p.I) + d_alpha[2]*Az(p.I) );
      });
}

extern "C" void LightThroughDM_RMSError(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_RMSError;
  DECLARE_CCTK_PARAMETERS;
  grid.loop_int_device<0, 0, 0>(
      grid.nghostzones,
      [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          using std::pow, std::sqrt;

          phi_RMSE(p.I) = sqrt( pow( phi(p.I) - phi_flat(p.I) ,2.0) );
          Ax_RMSE(p.I) = sqrt( pow( Ax(p.I) - Ax_flat(p.I) ,2.0) );
          Ay_RMSE(p.I) = sqrt( pow( Ay(p.I) - Ay_flat(p.I) ,2.0) );
          Az_RMSE(p.I) = sqrt( pow( Az(p.I) - Az_flat(p.I) ,2.0) );
                    
      });
}



} // namespace LightThroughDM