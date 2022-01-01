pub mod equil;
pub mod consts;
// pub mod rns_plotting;
mod equil_util;
use std::{process::exit, time::Instant,env};

use consts::*;
use equil::{get_e_p_surface, get_min_enthalpy, load_eos, make_center, make_grid};
use equil_util::{EosType, mass_radius, print_ns, sphere, spin};
use ndarray::Array2;
// use rns_plotting::write_array2;

// use crate::{equil_util::{RnsError, calc_sin_theta, print_header}, rns_plotting::contourf};
use crate::{equil_util::{RnsError, calc_sin_theta, print_header}};



fn main()  {
    if let Some(_manifest_dir) = env::var_os("CARGO_MANIFEST_DIR") {
        println!("Found CARGO_MANIFEST_DIR");
    } else {
        println!("Error finding CARGO_MANIFEST_DIR");
    }
    // let mut plot_timing = 0.0;
    let start = Instant::now();
    println!("starting");
    let (s,m) = make_grid();
    let (log_e_tab, log_p_tab, log_h_tab, log_n0_tab, _) = load_eos("./eos/eosA").unwrap(); 
            // LS_220_25-Sept-2017.rns
    let eos_type = &EosType::Table;
    // let opt_gamma_p = None;
    let opt_log_e_tab = &Some(log_e_tab);
    let opt_log_p_tab = &Some(log_p_tab);
    let opt_log_h_tab = &Some(log_h_tab);
    let opt_log_n0_tab = &Some(log_n0_tab);

    let enthalpy_min = get_min_enthalpy( eos_type);


    let (sin_theta, theta) = calc_sin_theta(&m);
    let mut sin_2n_p1_th_m = &mut Array2::<f64>::zeros((LMAX,MDIV)); 
    for n in 0 ..= LMAX-1 { // for(n=1;n<=LMAX;n++) {
        let scalar = 2.0 * n as f64 + 1.0;
        for m in 0 ..= MDIV-1 { // for(m=1;m<=MDIV-2;m+=2) {                       
            sin_2n_p1_th_m[[n,m]] = (scalar * theta[m]).sin();
        }
    }
    let rho = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let gama = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let alpha = &mut Array2::<f64>::zeros((SDIV, MDIV)); 
    let omega = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let energy = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let pressure = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let enthalpy = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let velocity_sq = &mut Array2::<f64>::zeros((SDIV, MDIV));
    let mut r_e = 0.0;

    println!();
    print_header();

    let base_e_center = 1e15;
    for e_scalar in 1 .. 2 {
        let mut continue_iteration = true;
        println!("\n*****************************************");
        println!("e_center = {:0.3}",base_e_center * e_scalar as f64 * CC * CC * KSCALE);
        println!("*****************************************\n");

        let unscaled_e_center = base_e_center * e_scalar as f64;
        let e_center = unscaled_e_center * CC * CC * KSCALE;
        
        let (p_center, h_center) = make_center(opt_log_e_tab, 
                                                opt_log_p_tab , 
                                                opt_log_h_tab, 
                                                &EosType::Table, 
                                                &None, 
                                                e_center).unwrap();

        let (e_surface, p_surface) = get_e_p_surface(eos_type);
        

        sphere(&s, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, None, 
                e_center, p_center, p_surface, e_surface, rho, gama, alpha, omega, &mut r_e);

        /* THE PROCEDURE SPIN() WILL COMPUTE THE METRIC OF A STAR WITH
            GIVEN OBLATENESS. THE OBLATENESS IS SPECIFIED BY GIVING 
            THE RATIO OF THE LENGTH OF THE AXIS CONNECTING THE CENTRE OF THE STAR 
            TO ONE OF THE POLES TO THE RADIUS OF THE STAR'S EQUATOR. 
            THIS RATIO IS NAMED r_ratio.
            WHEN r_ratio = 1.0, THE STAR IS SPHERICAL */            

        let mut r_ratio = 1.0;    
        let a_check = &mut 0;
        let accuracy =1e-5;
        let cf = 1.0;
        let big_omega = &mut 0.0;
  
        

        spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
            enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
            a_check, accuracy, cf, r_ratio, &mut r_e, big_omega, sin_theta, &mut sin_2n_p1_th_m);

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

        let res = mass_radius(
            &s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, rho, gama, alpha, omega, 
            energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, 
            r_e, mass, mass_0, ang_mom, rr_e, v_plus, v_minus, omega_k);
            
        // if let Err(error) = e { println!("Error, {}",error) };
        if let Err(e) = res { 
            println!("Invalid input parameters, continuing on next iteration. Error = {}",e);
            continue_iteration = false;
            continue;
        };
        // let start_of_first_plots = start.elapsed().as_secs_f64();
        // println!("Start initial plots");
        // contourf(&s, &m, rho, "metric potential: ρ", "./png/metric_rho"," Ω=0","_NonRotating");
        // contourf(&s, &m, gama, "metric potential: γ", "./png/metric_gamma"," Ω=0","_NonRotating");
        // contourf(&s, &m, alpha, "metric potential: α", "./png/metric_alpha"," Ω=0","_NonRotating");
        // contourf(&s, &m, omega, "metric potential associated with spin: ω", "./png/metric_omega"," Ω=0","_NonRotating");
        // contourf(&s, &m, energy, "energy", "./png/energy"," Ω=0","_NonRotating");
        // contourf(&s, &m, pressure, "pressure", "./png/pressure"," Ω=0","_NonRotating");
        // contourf(&s, &m, enthalpy, "enthalpy", "./png/enthalpy"," Ω=0","_NonRotating");
        // contourf(&s, &m, &velocity_sq.mapv(f64::sqrt), "abs(velocity)", "./png/abs_velocity"," Ω=0","_NonRotating");
        // println!("End initial plots");
        // let end_of_first_plots = start.elapsed().as_secs_f64();
        // plot_timing +=  end_of_first_plots - start_of_first_plots;
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


        while (diff_omega > 0.0) && continue_iteration {
            /* Find the interval of r_ratio where the star has the
                correct angular velocity	*/
            r_ratio -= dr;

            /* Compute the star with the specified value of r_ratio	*/

            spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
                enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
                a_check, accuracy, cf, r_ratio, &mut r_e, big_omega,sin_theta, &mut sin_2n_p1_th_m);

            let res = mass_radius(
                &s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, rho, gama, alpha, omega, 
                energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, 
                r_e, mass, mass_0, ang_mom, rr_e, v_plus, v_minus, omega_k);
                
            // if let Err(error) = e { println!("Error, {}",error) };
            if let Err(e) = res { 
                println!("Invalid input parameters, continuing on next iteration. Error = {}",e);
                continue_iteration = false;
                break;
            };

            print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);

            // if approx::abs_diff_eq!(r_ratio, 0.75) {
            //     let start_of_middle_plots = start.elapsed().as_secs_f64();
            //     println!("Start plotting");
            //     contourf(&s, &m, rho, "metric potential: ρ", "./png/metric_rho"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, gama, "metric potential: γ", "./png/metric_gamma"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, alpha, "metric potential: α", "./png/metric_alpha"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, omega, "metric potential associated with spin: ω", "./png/metric_omega"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, energy, "energy", "./png/energy"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, pressure, "pressure", "./png/pressure"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, enthalpy, "enthalpy", "./png/enthalpy"," ~Ωmax/2","_OmegamaxOn2");
            //     contourf(&s, &m, &velocity_sq.mapv(f64::sqrt), "abs(velocity)", "./png/abs_velocity"," ~Ωmax/2","_OmegamaxOn2");
            //     let end_of_middle_plots = start.elapsed().as_secs_f64();
            //     println!("End plotting");
            //     plot_timing += end_of_middle_plots - start_of_middle_plots;
            // }

            
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
        if continue_iteration {

            if (fl > 0.0 && fh < 0.0) || (fl < 0.0 && fh > 0.0) {
                let mut ans=-1.11e30;
                for j in 0 ..= 59 { //(j=1;j<=60;j++) {
                    let xm=0.5*(xl+xh);
                    r_ratio = xm;

                    spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
                        enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
                        a_check, accuracy, cf, r_ratio, &mut r_e, big_omega,sin_theta, &mut sin_2n_p1_th_m);

                    let res = mass_radius(
                        &s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, rho, gama, alpha, omega, 
                        energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, 
                        r_e, mass, mass_0, ang_mom, rr_e, v_plus, v_minus, omega_k);
                        
                    // if let Err(error) = e { println!("Error, {}",error) };
                    if let Err(e) = res { 
                        println!("Invalid input parameters, continuing on next iteration. Error = {}",e);
                        continue_iteration = false;
                        break;
                    };


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
                    ans=xnew;
                    r_ratio = ans;

                    spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
                        enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
                        a_check, accuracy, cf, r_ratio, &mut r_e, big_omega, sin_theta, &mut sin_2n_p1_th_m);

                    let res = mass_radius(
                        &s, &m, opt_log_e_tab, opt_log_n0_tab, eos_type, rho, gama, alpha, omega, 
                        energy, pressure, enthalpy, velocity_sq, r_ratio, e_surface, 
                        r_e, mass, mass_0, ang_mom, rr_e, v_plus, v_minus, omega_k);
                        
                    // if let Err(error) = e { println!("Error, {}",error) };
                    if let Err(e) = res { 
                        println!("Invalid input parameters, continuing on next iteration. Error = {}",e);
                        break;
                    };

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
                println!("\nInvalid combination of input parameters.");
                // continue;

                // panic!("Routine to calculate the final radius has failed, root not bracketed between r_ratio={} and r_ratio+dr={} as expected.", r_ratio, r_ratio + dr);
            }

            /* THE RIDDER ZERO-FINDING ROUTINE HAS FOUND THE VALUE
            OF R_RATIO WHICH GIVES THE DESIRED STAR. */



            
            print_header();
            print_ns(r_ratio, e_center, *mass, *mass_0, *rr_e, *big_omega, *omega_k, *ang_mom);
        } // if continue_iteration
    } // for

    // {   let start_of_final_plots = start.elapsed().as_secs_f64();
    //     println!("Start final plots");
    //     let rho: &Array2<f64> = &*rho;
    //     let gama: &Array2<f64> = &*gama;
    //     let alpha: &Array2<f64> = &*alpha;
    //     let omega: &Array2<f64> = &*omega;
    //     let energy: &Array2<f64> = &*energy;
    //     let pressure: &Array2<f64> = &*pressure;
    //     let enthalpy: &Array2<f64> = &*enthalpy;
    //     let velocity: &Array2<f64> = &velocity_sq.mapv(f64::sqrt);

    //     let filenames = ["rho", "gama", "alpha", "omega", 
    //                                 "energy", "pressure", 
    //                                 "enthalpy", "velocity"];

    //     let names = ["metric potential: ρ",
    //                             "metric potential: γ",
    //                             "metric potential: α",
    //                             "metric potential associated with spin: ω",
    //                             "energy",
    //                             "pressure",
    //                             "enthalpy",
    //                             "abs(velocity)"];


    //     for (idx,x) in [rho, gama, alpha, omega, energy, pressure, enthalpy, velocity].iter_mut().enumerate() {
    //         contourf(&s, &m, x, 
    //             names[idx], 
    //             &format!("./png/{}",filenames[idx]),
    //             " Ωmax",
    //             "_Omegamax");    
    //     }
    //     println!("End final plots");
    //     let end_of_final_plots = start.elapsed().as_secs_f64();
    //     plot_timing += end_of_final_plots - start_of_final_plots;
        
        
    //     // for (idx,  array) in [rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq].iter_mut().enumerate() {
    //         // let filename = format!("{}.csv",filenames[idx]);
    //         // write_array2(array,&filename[..]).unwrap();


    //     // contourf(&s, &m, rho, "metric potential: ρ", "./png/metric_rho"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, gama, "metric potential: γ", "./png/metric_gamma"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, alpha, "metric potential: α", "./png/metric_alpha"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, omega, "metric potential associated with spin: ω", "./png/metric_omega"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, energy, "energy", "./png/energy"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, pressure, "pressure", "./png/pressure"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, enthalpy, "enthalpy", "./png/enthalpy"," Ωmax","_Omegamax");
    //     // contourf(&s, &m, &velocity_sq.mapv(f64::sqrt), "abs(velocity)", "./png/abs_velocity"," Ωmax","_Omegamax");

    // }    

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n\nElapsed time = {:0.5}",elapsed);
    // println!("Plotting time = {:0.5}\n\n",plot_timing);
    // println!("Computational time = {:0.5}\n\n",elapsed - plot_timing);


    
}
