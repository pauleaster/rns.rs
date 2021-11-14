pub mod equil;
pub mod consts;
mod equil_util;
use std::time::Instant;

use consts::*;
use equil::{get_e_p_surface, get_min_enthalpy, load_eos, make_center, make_grid};
use equil_util::{EosType, mass_radius, print_ns, sphere, spin};
use ndarray::Array2;

use crate::equil_util::calc_sin_theta;




fn main()  {
    let start = Instant::now();
    let (s,m) = make_grid();
    let (log_e_tab, log_p_tab, log_h_tab, log_n0_tab, _) = load_eos("./eos/eosA").unwrap(); 
    let eos_type = &EosType::Table;
    // let opt_gamma_p = None;
    let opt_log_e_tab = &Some(log_e_tab);
    let opt_log_p_tab = &Some(log_p_tab);
    let opt_log_h_tab = &Some(log_h_tab);
    let opt_log_n0_tab = &Some(log_n0_tab);
    let unscaled_e_center = 1e15;
    let e_center = unscaled_e_center * CC * CC * KSCALE;
    
    let (p_center, h_center) = make_center(opt_log_e_tab, 
                                            opt_log_p_tab , 
                                            opt_log_h_tab, 
                                            &EosType::Table, 
                                            &None, 
                                            e_center).unwrap();

    let (e_surface, p_surface) = get_e_p_surface(eos_type);
    let mut rho = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let mut gama = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let mut alpha = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let mut omega = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut energy = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut pressure = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut enthalpy = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut velocity_sq = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut r_e = 0.0;

    sphere(&s, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, None, 
            e_center, p_center, p_surface, e_surface, &mut rho, &mut gama, &mut alpha, &mut omega, &mut r_e);

 /* THE PROCEDURE SPIN() WILL COMPUTE THE METRIC OF A STAR WITH
     GIVEN OBLATENESS. THE OBLATENESS IS SPECIFIED BY GIVING 
     THE RATIO OF THE LENGTH OF THE AXIS CONNECTING THE CENTRE OF THE STAR 
     TO ONE OF THE POLES TO THE RADIUS OF THE STAR'S EQUATOR. 
     THIS RATIO IS NAMED r_ratio.
     WHEN r_ratio = 1.0, THE STAR IS SPHERICAL */            

    let mut r_ratio = 1.0;
    let enthalpy_min = get_min_enthalpy( eos_type);
    let a_check = &mut 0;
    let accuracy =1e-5;
    let cf = 1.0;
    let big_omega = &mut 0.0;

    let (sin_theta, theta) = calc_sin_theta(&m);
    let mut sin_2n_p1_th_m = &mut Array2::<f64>::zeros((LMAX,MDIV)); 
    for n in 0 ..= LMAX-1 { // for(n=1;n<=LMAX;n++) {
        let scalar = 2.0 * n as f64 + 1.0;
        for m in 0 ..= MDIV-1 { // for(m=1;m<=MDIV-2;m+=2) {                       
            sin_2n_p1_th_m[[n,m]] = (scalar * theta[m]).sin();
        }
    }
    

    spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
        enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
        a_check, accuracy, cf, r_ratio, &mut r_e, big_omega, sin_theta, theta, &mut sin_2n_p1_th_m);

    /* THE METRIC FUNCTIONS ARE STORED IN THE FUNCTIONS 
        alpha, rho, gama, omega (see user's manual for the definition
        of the metric */


    /* COMPUTE THE VALUES OF VARIOUS EQUILIBRIUM QUANTITIES, SUCH
        AS MASS (mass), RADIUS (rr_e), BARYON MASS(mass_0), 
        ANGULAR MOMENTUM (ang_mom), 
        KEPLERIAN ANGULAR VELOCITY OF PARTICLE ORBITING AT 
        THE EQUATOR,
        VELOCITIES OF CO-ROTATING PARTICLES (v_plus),
        AND COUNTER-ROTATING PARTICLES (v_minus) */

    let mass: &mut f64 = &mut 0.0;
    let mass_0: &mut f64 = &mut 0.0;
    let ang_mom: &mut f64 = &mut 0.0;
    let rr_e: &mut f64 = &mut 0.0;
    let v_plus= &mut[0_f64;SDIV];
    let v_minus= &mut[0_f64;SDIV];
    let omega_k: &mut f64 = &mut 0.0;

    mass_radius(
        &s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, rho, gama, alpha, omega, 
        energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, 
        r_e, mass, mass_0, ang_mom, rr_e, v_plus, v_minus, omega_k);
    let elapsed = start.elapsed().as_secs_f64();

    print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);

    let dr = 0.05;

    /* THIS LOOP STARTS WITH A NON-ROTATING STAR AND INCREASES
        THE STAR'S OBLATENESS (BY DECREASING R_RATIO) AND 
        THEN CALCULATES THE STAR'S ANGULAR VELOCITY. ONCE THE
        COMPUTED VALUE OF ANGULAR VELOCITY IS LARGER THAN 
        THE ANGULAR VELOCITY OF A PARTICLE ORBITING THE STAR
        AT THE EQUATOR, (Omega_K), THE LOOP STOPS */

    let mut diff_omega = *omega_k - *big_omega;
    let mut old_diff_omega = diff_omega;

    // #[allow(clippy::never_loop)]
    while diff_omega > 0.0 {
        /* Find the interval of r_ratio where the star has the
            correct angular velocity	*/
        r_ratio -= dr;

        /* Compute the star with the specified value of r_ratio	*/

        spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
            enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
            a_check, accuracy, cf, r_ratio, &mut r_e, big_omega,sin_theta, theta, &mut sin_2n_p1_th_m);

        mass_radius(&s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, 
            rho, gama, alpha, omega, 
            energy, pressure, enthalpy, velocity_sq, 
            r_ratio, e_surface, r_e, mass, mass_0, ang_mom, 
            rr_e, v_plus, v_minus, omega_k);

        print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);

        
        old_diff_omega = diff_omega;
        diff_omega = *omega_k - *big_omega;
    }

        /* The correct star lies between r_ratio and r_ratio + dr */
    let mut xl = r_ratio;
    let mut xh = r_ratio + dr;
    let mut fl = diff_omega;
    let mut fh = old_diff_omega;

    /* Use Ridder's method to find the correct star (Taken from 
        Numerical Recipes zriddr())	*/

    let xacc = 1e-4;

    if (fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0) {
        let ans=-1.11e30;
        for j in 0 ..= 59 { //(j=1;j<=60;j++) {
            let xm=0.5*(xl+xh);
            r_ratio = xm;

            spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
                enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
                a_check, accuracy, cf, r_ratio, &mut r_e, big_omega,sin_theta, theta, &mut sin_2n_p1_th_m);

            mass_radius(&s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, 
                rho, gama, alpha, omega, 
                energy, pressure, enthalpy, velocity_sq, 
                r_ratio, e_surface, r_e, mass, mass_0, ang_mom, 
                rr_e, v_plus, v_minus, omega_k);


            let fm= *omega_k - *big_omega;
            let sroot=(fm*fm-fl*fh).sqrt();
            if approx::abs_diff_eq!(sroot, 0.0) {
                r_ratio = ans;
                break;
            }

            let xnew= xm + (xm-xl) * (fl-fh).signum() * fm / sroot;

            if (xnew-ans).abs() <= xacc {
                r_ratio = ans;
                break;
            }
            let ans=xnew;
            r_ratio = ans;

            spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
                enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
                a_check, accuracy, cf, r_ratio, &mut r_e, big_omega, sin_theta, theta, &mut sin_2n_p1_th_m);

            mass_radius(&s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, 
                rho, gama, alpha, omega, 
                energy, pressure, enthalpy, velocity_sq, 
                r_ratio, e_surface, r_e, mass, mass_0, ang_mom, 
                rr_e, v_plus, v_minus, omega_k);

            //   if(strcmp(eos_type,"tab")==0)
            // print(r_ratio,e_center, Mass, Mass_0, R_e, Omega, Omega_K, J);
            //   else 
            // printpoly(r_ratio,e_center, Mass, Mass_0, R_e, Omega, Omega_K, J);
            print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);


            let fnew =  *omega_k - *big_omega;
            if approx::abs_diff_eq!(fnew, 0.0) {
                r_ratio = ans;
                break;
            }
            if approx::abs_diff_ne!(fm.abs() * fnew.signum(),fm) { //(SIGN(fm,fnew) != fl) {
                xl=xm;
                fl=fm;
                xh=ans;
                fh=fnew;
            } else if approx::abs_diff_ne!(fl.abs() * fnew.signum(),fl) { //(SIGN(fl,fnew) != fl) {
                xh=ans;
                fh=fnew;
            } else if approx::abs_diff_ne!(fh.abs() * fnew.signum(),fh) { //(SIGN(fh,fnew) != fh) {
                xl=ans;
                fl=fnew;
            } else {
                panic!("Routine to calculate the final radius has failed.");
            }
            if (xh-xl).abs() <= xacc {
                r_ratio = ans;
                break;
            }
        }
    } 
    else {
        if approx::abs_diff_eq!(fh, 0.0) {
            r_ratio +=dr;
        }
        panic!("Routine to calculate the final radius has failed, root not bracketed between r_ratio{} and r_ratio+dr{} as expected.", r_ratio, r_ratio + dr);
    }

    /* THE RIDDER ZERO-FINDING ROUTINE HAS FOUND THE VALUE
    OF R_RATIO WHICH GIVES THE DESIRED STAR. */


    println!("\n\nElapsed time = {:0.5}",start.elapsed().as_secs_f64());
    print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);

}
