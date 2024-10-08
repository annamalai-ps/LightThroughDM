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
constexpr void plane_wave(const T lambdaC_prefactor, const T plane_wave_dist_from_DM, const T wavepacket_width, const T envelope_slope,
                             const T kx, const T ky, const T kz,
                             const T t, const T x, const T y, const T z,
                             T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi, T &alpha,
                             T &phi_flat, T &mu_flat, T &Ax_flat, T &nu_flat, T &Ay_flat, T &chi_flat, T &Az_flat, T &psi_flat) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt, std::erf, std::exp, std::abs;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));
  const T r_inv_cubed = pow((pow(x,2.0)+pow(y,2.0)+pow(z,2.0)),-1.5);
  const T r_square = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
  const T alpha_max = 0.1;
  const T M = pow(2.0*pow(pi, 3.0),0.25)*sqrt(lambdaC_prefactor*alpha_max);
  const T lambda = lambdaC_prefactor*(2*pi/M);
  //const T amp = exp(-pow((z-plane_wave_dist_from_DM)/wavepacket_width,2.0));
  
  /*-------Implement's wave train amplitude---------*/
  const T l1 = plane_wave_dist_from_DM;
  const T l2 = wavepacket_width + l1;
  double exp_term1 = 2 * envelope_slope * (z - l1);
  double exp_term2 = 2 * envelope_slope * (l2 - z);
  
  double amp = 0.0, d_amp =0.0, A_t1 = 0.0, A_t2 = 0.0, exp1 = 0.0, exp2 = 0.0;
  
  if (exp_term1 > 99) {
      d_amp = 0.0;
      A_t1 = 1.0;
  }
  if (exp_term1 < -99) {
      d_amp = 0.0;
      A_t1 = 0.0;
  }
  if (abs(exp_term1) < 99) {
      exp1 = exp(-exp_term1);
      A_t1 = 1 / (1 + exp1);
  }
  if (exp_term2 > 99) {
      d_amp = 0.0;
      A_t2 = 1.0;
  }
  if (exp_term2 < -99) {
      d_amp = 0.0;
      A_t2 = 0.0;
  }
  if (abs(exp_term2) < 99) {
      exp2 = exp(-exp_term2);
      A_t2 = 1 / (1 + exp2);
  }
  amp = A_t1 + A_t2 - 1;
  if (abs(exp_term1) < 99 && abs(exp_term2) < 99) {
      double d_At1 = exp1 * pow(A_t1, 2.0);
      double d_At2 = exp2 * pow(A_t2, 2.0);
      d_amp = 2.0 * envelope_slope * (d_At1 - d_At2);
  }
  /*-------------------------------------------*/
 

  if (x == 0 && y == 0 && z == 0.0){  //defn for indeterminate form at r=0
      alpha = M*pow(lambda,-1.0)*sqrt(2.0/pi);
  }
  else{
      alpha = M*pow(sqrt(r_square),-1.0)*erf(sqrt(r_square)/(sqrt(2.0)*lambda));
  }

  Ax = amp*cos(2.0*pi*(kz*z - t*omega));
  nu = d_amp*cos(2.0*pi*(kz*z - t*omega)) - amp*(2.0*pi*kz)*sin(2.0*pi*(kz*z - t*omega));
  Ay = 0.0;
  chi = 0.0;
  Az = 0.0;
  psi = 0.0;

  if (x == 0 && y == 0 && z == 0.0){  //defn forindeterminate form at r=0
    phi = 0.0;
    mu = 0.0;
  }
  else{
    phi = 0.0;//2.0*M*amp*pow(2.0*pi*omega,-1.0)*( x*sin(2.0*pi*(kz*z - t*omega)) - y*cos(2.0*pi*(kz*z - t*omega)) )*( erf(sqrt(r_square)/(sqrt(2)*lambda))*r_inv_cubed
        //- sqrt(2/pi)*exp(-r_square/(2*pow(lambda,2.0)))/(lambda*r_square) );

    mu = 2.0*M*(x*Ax + y*Ay + z*Az)*( erf(sqrt(r_square)/(sqrt(2)*lambda))*r_inv_cubed
        - sqrt(2.0/pi)*exp(-r_square/(2*pow(lambda,2.0)))/(lambda*r_square) );
  }

  Ax_flat = Ax;
  nu_flat = nu;
  Ay_flat = Ay;
  chi_flat = chi;
  Az_flat = Az;
  psi_flat = psi;
  phi_flat = 0.0;
  mu_flat = 0.0;

}

template <typename T>
constexpr void spline_alpha(const T lambdaC_prefactor, const T plane_wave_dist_from_DM, const T wavepacket_width,const T envelope_slope,
                             const T kx, const T ky, const T kz,
                             const T time, const T x, const T y, const T z,
                             T &phi, T &mu, T &Ax, T &nu, T &Ay, T &chi, T &Az, T &psi, T &alpha,
                             T &phi_flat, T &mu_flat, T &Ax_flat, T &nu_flat, T &Ay_flat, T &chi_flat, T &Az_flat, T &psi_flat) {
  using std::acos, std::cos, std::pow, std::sin, std::sqrt, std::erf, std::exp;

  const T pi = acos(-T(1));
  const T omega = sqrt(pow(kx, 2) + pow(ky, 2) + pow(kz, 2));
  const T r_square = pow(x, 2.0) + pow(y, 2.0) + pow(z, 2.0);
  const T alpha_max = 0.1;
  const T M = pow(2.0*pow(pi, 3.0),0.25)*sqrt(lambdaC_prefactor*alpha_max);
  const T lambda = lambdaC_prefactor*(2*pi/M);
  //const T amp = exp(-pow((z-plane_wave_dist_from_DM)/wavepacket_width,2.0));


  /*-------Implement's wave train amplitude---------*/
  const T l1 = plane_wave_dist_from_DM;
  const T l2 = wavepacket_width + l1;
  double exp_term1 = 2 * envelope_slope * (z - l1);
  double exp_term2 = 2 * envelope_slope * (l2 - z);
  
  double amp = 0.0, d_amp =0.0, A_t1 = 0.0, A_t2 = 0.0, exp1 = 0.0, exp2 = 0.0;
  
  if (exp_term1 > 99) {
      d_amp = 0.0;
      A_t1 = 1.0;
  }
  if (exp_term1 < -99) {
      d_amp = 0.0;
      A_t1 = 0.0;
  }
  if (exp_term2 > 99) {
      d_amp = 0.0;
      A_t2 = 1.0;
  }
  if (exp_term2 < -99) {
      d_amp = 0.0;
      A_t2 = 0.0; 
  }
  if (abs(exp_term1) < 99) {
      exp1 = exp(-exp_term1);
      A_t1 = 1 / (1 + exp1);
  }
  if (abs(exp_term2) < 99) {
      exp2 = exp(-exp_term2);
      A_t2 = 1 / (1 + exp2);
  }
  amp = A_t1 + A_t2 - 1;
  if (abs(exp_term1) < 99 && abs(exp_term2) < 99) {
      double d_At1 = exp1 * pow(A_t1, 2.0);
      double d_At2 = exp2 * pow(A_t2, 2.0);
      d_amp = 2.0 * envelope_slope * (d_At1 - d_At2);
  }
  /*-------------------------------------------*/

  const T r = sqrt(r_square);
  const T r1 = M/(0.12*alpha_max); // 12%
  const T r2 = M/(0.06*alpha_max); // 6%
  
  if (x == 0 && y == 0 && z == 0.0){  //defn for indeterminate form at r=0
    alpha = M*pow(lambda,-1.0)*sqrt(2.0/pi);
  }
  else if (r < r1) {
    alpha = M*pow(r,-1.0)*erf(r/(sqrt(2.0)*lambda));
  }
  else if (r<=r2 && r>=r1){
    const T dr = r2 - r1;
    const T t = (r - r1)/dr;
    const T alp1 = -M*pow(r1,-1.0)*erf(r1/(sqrt(2.0)*lambda));
    const T d_alp1 = (M*pow(r1,-2.0))*( erf(r1/(sqrt(2.0)*lambda)) - pow(lambda,-1.0)*sqrt(2/pi)*r1*exp(-(pow(r1,2.0)/(2*pow(lambda,2.0))) ) );
    const T alp2 = 0.0;
    const T d_alp2 = 0.0;
    const T a = d_alp1*dr - (alp2 - alp1);
    const T b = -d_alp2*dr + (alp2 - alp1);
    
    alpha = -(1-t)*alp1 + t*alp2 + t*(1-t)*( a*(1-t) - b*t );
  }
  if (r>r2){
    alpha = 0.0;
  }

  Ax = amp*cos(2.0*pi*(kz*z - omega*time));
  nu = d_amp*cos(2.0*pi*(kz*z - omega*time)) - amp*(2.0*pi*kz)*sin(2.0*pi*(kz*z - omega*time));
  Ay = amp*sin(2.0*pi*(kz*z - omega*time));
  chi = d_amp*sin(2.0*pi*(kz*z - omega*time)) + amp*(2.0*pi*kz)*cos(2.0*pi*(kz*z - omega*time));
  Az = 0.0;
  psi = 0.0;
  phi = 0.0;
  mu = 0.0;

  Ax_flat = Ax;
  nu_flat = nu;
  Ay_flat = Ay;
  chi_flat = chi;
  Az_flat = Az;
  psi_flat = psi;
  phi_flat = 0.0;
  mu_flat = 0.0;

}

extern "C" void LightThroughDM_Initial(CCTK_ARGUMENTS) {
  DECLARE_CCTK_ARGUMENTSX_LightThroughDM_Initial;
  DECLARE_CCTK_PARAMETERS;
    
    grid.loop_int_device<0, 0, 0>(
        grid.nghostzones,
        [=] CCTK_DEVICE(const Loop::PointDesc &p) CCTK_ATTRIBUTE_ALWAYS_INLINE {
          
          if (CCTK_EQUALS(initial_condition, "plane wave")) {
            plane_wave(lambdaC_prefactor, plane_wave_dist_from_DM, wavepacket_width, envelope_slope,
                          plane_wave_kx, plane_wave_ky, plane_wave_kz,
                          cctk_time, p.x, p.y, p.z,
                          phi(p.I), mu(p.I), Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I), alpha(p.I),
                          phi_flat(p.I), mu_flat(p.I), Ax_flat(p.I), nu_flat(p.I), Ay_flat(p.I), chi_flat(p.I), Az_flat(p.I), psi_flat(p.I));
            
          }
          else if (CCTK_EQUALS(initial_condition, "spline_alpha")) {
            spline_alpha(lambdaC_prefactor, plane_wave_dist_from_DM, wavepacket_width, envelope_slope,
                          plane_wave_kx, plane_wave_ky, plane_wave_kz,
                          cctk_time, p.x, p.y, p.z,
                          phi(p.I), mu(p.I), Ax(p.I), nu(p.I), Ay(p.I), chi(p.I), Az(p.I), psi(p.I), alpha(p.I),
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
          Arith::vect<CCTK_REAL, dim> d_alpha;
          Arith::vect<CCTK_REAL, dim> dd_alpha;

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

            d_alpha[d] = (-alpha(p.I + 2*p.DI[d]) + 8.0*alpha(p.I + p.DI[d]) -8.0*alpha(p.I - p.DI[d])
                     + alpha(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
            dd_alpha[d] = ( -alpha(p.I + 2*p.DI[d]) + 16.0*alpha(p.I + p.DI[d]) - 30.0*alpha(p.I) + 
                      16.0*alpha(p.I - p.DI[d]) - alpha(p.I - 2*p.DI[d]) ) / ( 12.0*pow(p.DX[d], 2.0) );
          
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


          phi_rhs(p.I) = mu(p.I);
          mu_rhs(p.I) = pow(1 + 2.0*alpha(p.I),-1.0)*( (1 - 2.0*alpha(p.I))*(dd_phi[0] + dd_phi[1] + dd_phi[2])
                        - 2.0*phi(p.I)*(dd_alpha[0]+dd_alpha[1]+dd_alpha[2])
                        + 2.0*(d_alpha[0]*nu(p.I) + d_alpha[1]*chi(p.I) + d_alpha[2]*psi(p.I))
                        - 2.0*(d_alpha[0]*d_phi[0] + d_alpha[1]*d_phi[1] + d_alpha[2]*d_phi[2]) );
          Ax_rhs(p.I) = nu(p.I);
          nu_rhs(p.I) = pow(1 + 2.0*alpha(p.I),-1.0)*( (1 - 2.0*alpha(p.I))*(dd_Ax[0] + dd_Ax[1] + dd_Ax[2])
                        + 2.0*Ax(p.I)*(dd_alpha[0]+dd_alpha[1]+dd_alpha[2])
                        + 2.0*(d_alpha[2]*d_Az[0] + d_alpha[1]*d_Ay[0])
                        - 2.0*d_alpha[0]*(-mu(p.I) + d_Ay[1] + d_Az[2])
                        + 2.0*(d_alpha[0]*d_Ax[0] + d_alpha[1]*d_Ax[1] + d_alpha[2]*d_Ax[2]) );
          Ay_rhs(p.I) = chi(p.I);
          chi_rhs(p.I) = pow(1 + 2.0*alpha(p.I),-1.0)*( (1 - 2.0*alpha(p.I))*(dd_Ay[0] + dd_Ay[1] + dd_Ay[2])
                        + 2.0*Ay(p.I)*(dd_alpha[0]+dd_alpha[1]+dd_alpha[2])
                        + 2.0*(d_alpha[0]*d_Ax[1] + d_alpha[2]*d_Az[1])
                        - 2.0*d_alpha[1]*(-mu(p.I) + d_Ax[0] + d_Az[2])
                        + 2.0*(d_alpha[0]*d_Ay[0] + d_alpha[1]*d_Ay[1] + d_alpha[2]*d_Ay[2]) );  
          Az_rhs(p.I) = psi(p.I);
          psi_rhs(p.I) = pow(1 + 2.0*alpha(p.I),-1.0)*( (1 - 2.0*alpha(p.I))*(dd_Az[0] + dd_Az[1] + dd_Az[2])
                        + 2.0*Az(p.I)*(dd_alpha[0]+dd_alpha[1]+dd_alpha[2])
                        + 2.0*(d_alpha[0]*d_Ax[2] + d_alpha[1]*d_Ay[2])
                        - 2.0*d_alpha[2]*(-mu(p.I) + d_Ax[0] + d_Ay[1])
                        + 2.0*(d_alpha[0]*d_Az[0] + d_alpha[1]*d_Az[1] + d_alpha[2]*d_Az[2]) );
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
          Arith::vect<CCTK_REAL, dim> d_alpha;

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

            d_alpha[d] = (-alpha(p.I + 2*p.DI[d]) + 8.0*alpha(p.I + p.DI[d]) -8.0*alpha(p.I - p.DI[d])
                     + alpha(p.I - 2*p.DI[d]) )/(12.0*p.DX[d]);
          }

          // for flat spacetime
          constraint_violation_flat(p.I) = ( mu_flat(p.I) + d_Ax_flat[0] + d_Ay_flat[1] + d_Az_flat[2] );
          
      
          constraint_violation(p.I) = ( mu(p.I) + d_Ax[0] + d_Ay[1] + d_Az[2] ) 
                                      + 2.0*(d_alpha[0]*Ax(p.I) + d_alpha[1]*Ay(p.I) + d_alpha[2]*Az(p.I) );
      });
}

/*
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
*/


} // namespace LightThroughDM
