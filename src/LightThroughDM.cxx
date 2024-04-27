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
constexpr void plane_wave(const T M, const T a_ext, const T A, const T kx, const T ky, const T kz,
                             const T t, const T x, const T y, const T z,
                             T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi,
                             T &phi_flat, T &mu_flat, T &Ax_flat, T &nu_flat, T &Ay_flat, T &chi_flat, T &Az_flat, T &psi_flat) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));
  const T r_inv_cubed = pow((pow(x,2.0)+pow(y,2.0)+pow(z,2.0)),-1.5);
  const T r_square = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);

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

    Ax_flat = Ax;
    nu_flat = nu;
    Ay_flat = Ay;
    chi_flat = chi;
    Az_flat = Az;
    psi_flat = psi;

    phi_flat = 0.0;
    mu_flat = 0.0;

    if (sqrt(r_square) >a_ext) // exterior
    {
      phi = M*A*r_inv_cubed*pow(pi*omega,-1.0)*( x*sin(2*pi*omega*(z + t)) - y*cos(2*pi*omega*(z + t)) );
      mu = 2.0*M*A*r_inv_cubed*( x*cos(2*pi*omega*(z + t)) + y*sin(2*pi*omega*(z + t))  );
    }
    else //interior
    {
      phi = M*A*pow(a_ext,-3.0)*pow(pi*omega,-1.0)*( x*sin(2*pi*omega*(z + t)) - y*cos(2*pi*omega*(z + t)) );
      mu = 2.0*M*A*pow(a_ext,-3.0)*( x*cos(2*pi*omega*(z + t)) + y*sin(2*pi*omega*(z + t))  );
    }
  }
  else
  {
    phi = 0.0, mu = 0.0, Ax = 0.0, nu = 0.0, Ay = 0.0, chi = 0.0, Az = 0.0, psi = 0.0;
    phi_flat = 0.0, mu_flat = 0.0, Ax_flat = 0.0, nu_flat = 0.0, Ay_flat = 0.0, chi_flat = 0.0, Az_flat = 0.0, psi_flat = 0.0;
  }

}

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
    
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          
          if (CCTK_EQUALS(initial_condition, "plane wave")) {
            plane_wave(M, a_ext, amplitude, standing_wave_kx, standing_wave_ky,
                          standing_wave_kz, cctk_time, p.x, p.y, p.z,
                          phi(p.I), mu(p.I), Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I),
                          phi_flat(p.I), mu_flat(p.I), Ax_flat(p.I), nu_flat(p.I), Ay_flat(p.I), chi_flat(p.I), Az_flat(p.I), psi_flat(p.I));
            
          }
          else if(CCTK_EQUALS(initial_condition, "standing wave")) {
            standing_wave(amplitude, standing_wave_kx, standing_wave_ky,
                          standing_wave_kz, cctk_time, p.x, p.y, p.z, phi(p.I), mu(p.I),
                          Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I));
          }
          else if (CCTK_EQUALS(initial_condition, "Gaussian")) {
            gaussian(amplitude, gaussian_width, gaussian_x_offset, cctk_time, p.x, p.y, p.z,  phi(p.I), mu(p.I),
                    Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I));
          }
          else {
            CCTK_ERROR("Unknown initial condition");
          }
    
          using std::pow, std::sqrt;
          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0); 

          if (sqrt(r_square) >a_ext) // exterior
          {
            density(p.I) = 0.0;
            pressure(p.I) = 0.0;
          }
          else //interior
          {
            CCTK_REAL constC = sqrt(M/(2.0*pow(a_ext,3.0)));
            CCTK_REAL constA = constC*(4.0*a_ext - M)/(2.0*a_ext + M);
            CCTK_REAL constB = (1.0/constC)*(2.0*a_ext - 2.0*M)/(2.0*a_ext + M);
            CCTK_REAL rho_int = (3.0*M)/((4.0*M_PI*pow(a_ext,3.0))*pow(1 + M/(2.0*a_ext),6.0));
            CCTK_REAL R2 = 3.0/(8.0*M_PI*rho_int);
            CCTK_REAL P_int = ( constA*( pow(constC,-2.0) - 2.0*r_square ) 
                              + constB*(r_square*pow(constC,2.0) - 2.0) )/( 8.0*M_PI*R2*(constA*r_square + constB) );
            density(p.I) = rho_int;
            pressure(p.I) = P_int;
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
          using std::pow, std::sqrt;

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

          Arith::vect<CCTK_REAL, dim> d4_phi;
          Arith::vect<CCTK_REAL, dim> d4_Ax;
          Arith::vect<CCTK_REAL, dim> d4_Ay;
          Arith::vect<CCTK_REAL, dim> d4_Az;
          Arith::vect<CCTK_REAL, dim> d4_mu;
          Arith::vect<CCTK_REAL, dim> d4_nu;
          Arith::vect<CCTK_REAL, dim> d4_chi;
          Arith::vect<CCTK_REAL, dim> d4_psi;


          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0);

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

            // dd_mu[d] = ( -mu(p.I + 2*p.DI[d]) + 16.0*mu(p.I + p.DI[d]) - 30.0*mu(p.I) + 
            //           16.0*mu(p.I - p.DI[d]) - mu(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            // dd_nu[d] = ( -nu(p.I + 2*p.DI[d]) + 16.0*nu(p.I + p.DI[d]) - 30.0*nu(p.I) + 
            //           16.0*nu(p.I - p.DI[d]) - nu(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            // dd_chi[d] = ( -chi(p.I + 2*p.DI[d]) + 16.0*chi(p.I + p.DI[d]) - 30.0*chi(p.I) + 
            //           16.0*chi(p.I - p.DI[d]) - chi(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            // dd_psi[d] = ( -psi(p.I + 2*p.DI[d]) + 16.0*psi(p.I + p.DI[d]) - 30.0*psi(p.I) + 
            //           16.0*psi(p.I - p.DI[d]) - psi(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            
            
            dd_phi_flat[d] = ( -phi_flat(p.I + 2*p.DI[d]) + 16.0*phi_flat(p.I + p.DI[d]) - 30.0*phi_flat(p.I) + 
                      16.0*phi_flat(p.I - p.DI[d]) - phi_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ax_flat[d] = ( -Ax_flat(p.I + 2*p.DI[d]) + 16.0*Ax_flat(p.I + p.DI[d]) - 30.0*Ax_flat(p.I) + 
                      16.0*Ax_flat(p.I - p.DI[d]) - Ax_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Ay_flat[d] = ( -Ay_flat(p.I + 2*p.DI[d]) + 16.0*Ay_flat(p.I + p.DI[d]) - 30.0*Ay_flat(p.I) + 
                      16.0*Ay_flat(p.I - p.DI[d]) - Ay_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
            dd_Az_flat[d] = ( -Az_flat(p.I + 2*p.DI[d]) + 16.0*Az_flat(p.I + p.DI[d]) - 30.0*Az_flat(p.I) + 
                      16.0*Az_flat(p.I - p.DI[d]) - Az_flat(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );

            //fourth derivative
            d4_phi[d] = ( phi(p.I + 2.0*p.DI[d]) - 4.0*phi(p.I + p.DI[d]) + 6.0*phi(p.I) 
                          - 4.0*phi(p.I - p.DI[d]) + phi(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_Ax[d] = ( Ax(p.I + 2.0*p.DI[d]) - 4.0*Ax(p.I + p.DI[d]) + 6.0*Ax(p.I) 
                          - 4.0*Ax(p.I - p.DI[d]) + Ax(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_Ay[d] = ( Ay(p.I + 2.0*p.DI[d]) - 4.0*Ay(p.I + p.DI[d]) + 6.0*Ay(p.I) 
                          - 4.0*Ay(p.I - p.DI[d]) + Ay(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_Az[d] = ( Az(p.I + 2.0*p.DI[d]) - 4.0*Az(p.I + p.DI[d]) + 6.0*Az(p.I) 
                          - 4.0*Az(p.I - p.DI[d]) + Az(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_mu[d] = ( mu(p.I + 2.0*p.DI[d]) - 4.0*mu(p.I + p.DI[d]) + 6.0*mu(p.I) 
                          - 4.0*mu(p.I - p.DI[d]) + mu(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_nu[d] = ( nu(p.I + 2.0*p.DI[d]) - 4.0*nu(p.I + p.DI[d]) + 6.0*nu(p.I) 
                          - 4.0*nu(p.I - p.DI[d]) + nu(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_chi[d] = ( chi(p.I + 2.0*p.DI[d]) - 4.0*chi(p.I + p.DI[d]) + 6.0*chi(p.I) 
                          - 4.0*chi(p.I - p.DI[d]) + chi(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
            d4_psi[d] = ( psi(p.I + 2.0*p.DI[d]) - 4.0*psi(p.I + p.DI[d]) + 6.0*psi(p.I) 
                          - 4.0*psi(p.I - p.DI[d]) + psi(p.I - 2.0*p.DI[d]) )/pow(p.DX[d],4.0);
          
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

          if (sqrt(r_square) >a_ext) // exterior
          {
            const CCTK_REAL alpha_ext = M / sqrt(r_square);
            Arith::vect<CCTK_REAL, dim> d_alpha_ext;
            Arith::vect<CCTK_REAL, dim> dd_alpha_ext;

            for (int d = 0; d < dim; ++d) {
              d_alpha_ext[d] = -(M*p.X[d]) / pow(r_square,1.5);
              dd_alpha_ext[d] = M*((3.0*pow(p.X[d], 2.0)) / pow(r_square,2.5) - 1.0/pow(r_square,1.5)); 
            }

            phi_rhs(p.I) = mu(p.I);
            mu_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                          -phi(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*nu(p.I) + d_alpha_ext[1]*chi(p.I) + d_alpha_ext[2]*psi(p.I))
                          - 2.0*(d_alpha_ext[0]*d_phi[0] + d_alpha_ext[1]*d_phi[1] + d_alpha_ext[2]*d_phi[2])
                          + 4.0*M_PI*(density(p.I) + 3.0*pressure(p.I))*phi(p.I) );
            Ax_rhs(p.I) = nu(p.I);
            nu_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                          +Ax(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[2]*d_Az[0] + d_alpha_ext[1]*d_Ay[0])
                          - 2.0*d_alpha_ext[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[0] + d_alpha_ext[1]*d_Ax[1] + d_alpha_ext[2]*d_Ax[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Ax(p.I) );
            Ay_rhs(p.I) = chi(p.I);
            chi_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                          +Ay(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[1] + d_alpha_ext[2]*d_Az[1])
                          - 2.0*d_alpha_ext[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                          + 2.0*(d_alpha_ext[0]*d_Ay[0] + d_alpha_ext[1]*d_Ay[1] + d_alpha_ext[2]*d_Ay[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Ay(p.I) );  
            Az_rhs(p.I) = psi(p.I);
            psi_rhs(p.I) = pow(1 + 2.0*alpha_ext,-1.0)*( (1 - 2.0*alpha_ext)*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                          +Az(p.I)*(dd_alpha_ext[0] + dd_alpha_ext[1] + dd_alpha_ext[2])
                          + 2.0*(d_alpha_ext[0]*d_Ax[2] + d_alpha_ext[1]*d_Ay[2])
                          - 2.0*d_alpha_ext[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                          + 2.0*(d_alpha_ext[0]*d_Az[0] + d_alpha_ext[1]*d_Az[1] + d_alpha_ext[2]*d_Az[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Az(p.I) );

          }
          else //interior
          {

            const CCTK_REAL alpha_int = ( M / 2.0 * a_ext )*(3.0 - r_square/pow(a_ext, 2.0) );
            const CCTK_REAL dd_alpha_int = M/pow(a_ext, 3.0);
            Arith::vect<CCTK_REAL, dim> d_alpha_int;
            for (int d = 0; d < dim; ++d)
              d_alpha_int[d] = -(M*p.X[d]) / pow(a_ext, 3.0);

            phi_rhs(p.I) = mu(p.I);
            mu_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                          -phi(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*nu(p.I) + d_alpha_int[1]*chi(p.I) + d_alpha_int[2]*psi(p.I))
                          - 2.0*(d_alpha_int[0]*d_phi[0] + d_alpha_int[1]*d_phi[1] + d_alpha_int[2]*d_phi[2])
                          + 4.0*M_PI*(density(p.I) + 3.0*pressure(p.I))*phi(p.I) );
            Ax_rhs(p.I) = nu(p.I);
            nu_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                          +Ax(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[2]*d_Az[0] + d_alpha_int[1]*d_Ay[0])
                          - 2.0*d_alpha_int[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                          + 2.0*(d_alpha_int[0]*d_Ax[0] + d_alpha_int[1]*d_Ax[1] + d_alpha_int[2]*d_Ax[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Ax(p.I) );
            Ay_rhs(p.I) = chi(p.I);
            chi_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                          +Ay(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*d_Ax[1] + d_alpha_int[2]*d_Az[1])
                          - 2.0*d_alpha_int[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                          + 2.0*(d_alpha_int[0]*d_Ay[0] + d_alpha_int[1]*d_Ay[1] + d_alpha_int[2]*d_Ay[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Ay(p.I) );  
            Az_rhs(p.I) = psi(p.I);
            psi_rhs(p.I) = pow(1 + 2.0*alpha_int,-1.0)*( (1 - 2.0*alpha_int)*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                          +Az(p.I)*(3*dd_alpha_int)
                          + 2.0*(d_alpha_int[0]*d_Ax[2] + d_alpha_int[1]*d_Ay[2])
                          - 2.0*d_alpha_int[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                          + 2.0*(d_alpha_int[0]*d_Az[0] + d_alpha_int[1]*d_Az[1] + d_alpha_int[2]*d_Az[2])
                          - 4.0*M_PI*(density(p.I) - pressure(p.I))*Az(p.I) );            

          }
          // First order Kreiss-Oliger dissipation
          // phi_rhs(p.I) = phi_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_phi[0] + dd_phi[1] + dd_phi[2]);
          // mu_rhs(p.I) = mu_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_mu[0] + dd_mu[1] + dd_mu[2]);
          // Ax_rhs(p.I) = Ax_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2]);
          // nu_rhs(p.I) = nu_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_nu[0] + dd_nu[1] + dd_nu[2]);
          // Ay_rhs(p.I) = Ay_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2]);
          // chi_rhs(p.I) = chi_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_chi[0] + dd_chi[1] + dd_chi[2]);
          // Az_rhs(p.I) = Az_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_Az[0] + dd_Az[1] + dd_Az[2]);
          // psi_rhs(p.I) = psi_rhs(p.I) + (epsdiss*p.DX[0]/4.0)*(dd_psi[0] + dd_psi[1] + dd_psi[2]);

          // Third order Kreiss-Oliger dissipation
          phi_rhs(p.I) = phi_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_phi[0] + d4_phi[1] + d4_phi[2]);
          mu_rhs(p.I) = mu_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_mu[0] + d4_mu[1] + d4_mu[2]);
          Ax_rhs(p.I) = Ax_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_Ax[0] + d4_Ax[1] + d4_Ax[2]);
          nu_rhs(p.I) = nu_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_nu[0] + d4_nu[1] + d4_nu[2]);
          Ay_rhs(p.I) = Ay_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_Ay[0] + d4_Ay[1] + d4_Ay[2]);
          chi_rhs(p.I) = chi_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_chi[0] + d4_chi[1] + d4_chi[2]);
          Az_rhs(p.I) = Az_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_Az[0] + d4_Az[1] + d4_Az[2]);
          psi_rhs(p.I) = psi_rhs(p.I) - (epsdiss*pow(p.DX[0],3.0)/16.0)*(d4_psi[0] + d4_psi[1] + d4_psi[2]);

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
          using std::pow, std::sqrt;

          Arith::vect<CCTK_REAL, dim> d_phi;
          Arith::vect<CCTK_REAL, dim> d_Ax;
          Arith::vect<CCTK_REAL, dim> d_Ay;
          Arith::vect<CCTK_REAL, dim> d_Az;
          Arith::vect<CCTK_REAL, dim> d_phi_flat;
          Arith::vect<CCTK_REAL, dim> d_Ax_flat;
          Arith::vect<CCTK_REAL, dim> d_Ay_flat;
          Arith::vect<CCTK_REAL, dim> d_Az_flat;
          const CCTK_REAL r_square = pow(p.x, 2.0) + pow(p.y, 2.0) + pow(p.z, 2.0); 

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

          if (sqrt(r_square) >a_ext) // exterior
          {
            // const CCTK_REAL alpha_ext = M / sqrt(r_square);
            Arith::vect<CCTK_REAL, dim> d_alpha_ext;

            for (int d = 0; d < dim; ++d) {
              d_alpha_ext[d] = -(M*p.X[d]) / pow(r_square,1.5);
            }
        
            constraint_violation(p.I) = ( mu(p.I) + d_Ax[0] + d_Ay[1] + d_Az[2] ) 
                                        + 2.0*(d_alpha_ext[0]*Ax(p.I) + d_alpha_ext[1]*Ay(p.I) + d_alpha_ext[2]*Az(p.I) );
          }            
          else //interior
          {            
            // const CCTK_REAL alpha_int = ( M / 2.0 * a_ext )*(3.0 - r_square/pow(a_ext, 2.0) );
            Arith::vect<CCTK_REAL, dim> d_alpha_int;
            for (int d = 0; d < dim; ++d)
              d_alpha_int[d] = -(M*p.X[d]) / pow(a_ext, 3.0);
            
            constraint_violation(p.I) = ( mu(p.I) + d_Ax[0] + d_Ay[1] + d_Az[2] ) 
                                        + 2.0*(d_alpha_int[0]*Ax(p.I) + d_alpha_int[1]*Ay(p.I) + d_alpha_int[2]*Az(p.I) );
          }
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
