use std::cmp::{min,max};
use std::f64::consts::PI;
use assert_approx_eq::assert_approx_eq;
use ndarray::{Array2, array};
use std::error::Error;

use crate::consts::*;
use crate::equil::{e_at_p, e_of_rho0, load_eos, make_grid, read_eos_file};


pub enum EosType {
    Table,
    Polytropic,
}

#[test]
fn test_hunt() {
    
    let x1:Vec<f64> = (0 ..20).map(f64::from).collect();
    let y1:Vec<f64> = (0 ..20).map(|x| (x as f64).powi(2)).collect();
    println!("{:?}", &x1);
    println!("{:?}", &y1);
    for ref_val in &[0.1_f64, 5.9, 11.0, 12.2, 17.9, 18.2] {
        let result_val = ref_val.floor() as usize;
        let closest = ref_val.round() as usize;
        let y_result_val = result_val.pow(2);
        let y_closest = closest.pow(2);
        let y_ref_val = ref_val.powi(2);
        
        
        println!("ref_val={} , expected result_val={}, closest={}", ref_val, result_val, closest);
        println!("y_ref_val={} , expected y_result_val={}, y_closest={}", y_ref_val, y_result_val, y_closest);
        let z = hunt(&y1, y_ref_val, Some(3));
        println!("z={}",z);
        println!("{} == {}? = {}",result_val, z, result_val ==z );
        assert!((z as f64 - result_val as f64).abs() <= 1.0);
    }
    let y1:Vec<f64> = (0 ..20).map(|x| 400. - (x as f64).powi(2)).collect();
    println!("{:?}", &x1);
    println!("{:?}", &y1);
    for ref_val in &[0.1_f64, 5.9, 11.0, 12.2, 17.9, 18.2] {
        let result_val = ref_val.floor() as usize;
        let closest = ref_val.round() as usize;
        let y_result_val = 400 - result_val.pow(2);
        let y_closest = 400 -closest.pow(2);
        let y_ref_val = 400. - ref_val.powi(2);
        
        
        println!("ref_val={} , expected result_val={}, closest={}", ref_val, result_val, closest);
        println!("y_ref_val={} , expected y_result_val={}, y_closest={}", y_ref_val, y_result_val, y_closest);
        let z = hunt(&y1, y_ref_val, None);
        println!("z={}",z);
        println!("{} == {}? = {}",result_val, z, result_val ==z );
        assert!((z as f64 - result_val as f64).abs() <= 1.0);
    }

}

/***************************************************************************/
/* Routine that locates nearest grid point for a given value.              */
/* Adapted from Numerical Recipes.                                         */
/***************************************************************************/
fn hunt(xx: &[f64], x: f64, opt_jlo: Option<usize>) -> usize {

	let n = xx.len();
    let mut jlo = match opt_jlo {
        Some(j) => j,
        None => n >> 1,
    };

    
    let mut jhi;
	let ascnd = xx[n - 1] > xx[0];
    let mut jm;
	
    let mut inc = 1;
    if (x >= xx[jlo]) == ascnd {
        if jlo == n - 1 {
            return jlo;
        } 
        jhi = jlo + 1;
        while (x >= xx[jhi]) == ascnd {
            jlo = jhi;
            inc <<= 1;
            jhi = jlo + inc;
            if  jhi > n - 1  {
                jhi = n - 1;
                break;
            }
        }
    } else {
        if  jlo == 0  {
            jlo = 0;
            return jlo;
        }
        jhi = jlo;
        jlo -= 1;
        while (x < xx[jlo]) == ascnd {
            jhi = jlo ;
            inc += inc << 1;
            jlo = max(jhi as i32 - inc as i32, 0) as usize;
            if  jlo == 0  {
                break;
            }
        }
    }
	
	while  jhi - jlo  != 1 {
		jm = (jhi + jlo) >> 1;
		if (x > xx[jm]) == ascnd {
			jlo = jm;
        }
		else {
			jhi = jm;
        }
	}
    jlo
}

#[test]
fn test_round_from_zero() {
    let scale = 1e10;
    let xvals = &[0.5 * f64::EPSILON, -0.5 * f64::EPSILON, 1.5 * f64::EPSILON, -1.5 * f64::EPSILON];
    let results = &[f64::EPSILON, f64::EPSILON, 1.5 * f64::EPSILON, -1.5 * f64::EPSILON];
    for (&x, &y) in xvals.iter().zip(results) {
        assert_approx_eq!(round_from_zero(x) * scale, y * scale);
    }
}

fn round_from_zero (x : f64) -> f64 {

    match x.abs() < f64::EPSILON {
        true => f64::EPSILON,
        false => x,
    }
}

#[test]
fn test_interp() {

    let xp = &[0.1_f64, 0.5,     1.0, 2.0,  2.5,  2.6,  3.0];
    let yp = &[0.1_f64, 0.1, 0.2, 0.25, 0.26, 2.65, 2.7];

    let xbvals = [0.3_f64, 1.5, 2.8];
    let results = [0.086_362_573_099_415_2_f64, 0.240_000_000_000_000_0, 5.625_200_000_000_000_0];
    // let kvals = [0_usize, 1, 3];
    
    for (idx, &xbval) in xbvals.iter().enumerate() {
        let yb = interp(xp, yp, xbval, Some(6));
        assert_approx_eq!(yb, results[idx]);
    }
}

/*************************************************************************/
/* Driver for the interpolation routine. First we find the tab. point    */
/* nearest to xb, then we interpolate using four points around xb.       */  
/* This is a direct interpretation of polint from numerical recipes      */  
/* Using equation 3.1.1.                                                 */  
/*************************************************************************/
pub fn interp(  xp: &[f64], 
            yp: &[f64], 
            xb: f64,
            opt_nearest: Option<usize>) -> f64 { 

    let nearest = hunt(xp,xb, opt_nearest);

    let np = xp.len();
    let k=max(0,
                    min(nearest as i32 - 1, np as i32 - 4)) as usize;

    // epsilon shift corrected, should eliminate (xp[i]-xp[j]).abs() < eps


    let d1 = round_from_zero((xp[k]-xp[k+1])*(xp[k]-xp[k+2])*(xp[k]-xp[k+3]));
    let d2 = round_from_zero((xp[k+1]-xp[k])*(xp[k+1]-xp[k+2])*(xp[k+1]-xp[k+3]));
    let d3 = round_from_zero((xp[k+2]-xp[k])*(xp[k+2]-xp[k+1])*(xp[k+2]-xp[k+3]));
    let d4 = round_from_zero((xp[k+3]-xp[k])*(xp[k+3]-xp[k+1])*(xp[k+3]-xp[k+2]));


    (xb-xp[k+1])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k]/ d1
        + (xb-xp[k])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k+1]/ d2 
        + (xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+3])*yp[k+2]/ d3
        + (xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+2])*yp[k+3]/ d4

    
}

#[test]
fn test_rtsec_g() {
    let e_center = 61.1558;
    let gamma = 2.2639; 
    let rho_center = 6.4936;
    let rho0_center_est = rtsec_g( &e_of_rho0, gamma, 0.0,e_center,f64::EPSILON, 
        e_center ).unwrap();
    println!("rho_center = {}, rho0_center_est = {}", rho_center, rho0_center_est);
    assert_approx_eq!(rho_center, rho0_center_est, 0.0001);
}

#[test]
#[should_panic]
fn test_rtsec_g_fail() {
    let e_center = 12.0;
    let gamma = 2.2639; // gamma estimate at the center
    let rho_center = 6.4936;
    let rho0_center_est = rtsec_g( &e_of_rho0, gamma, 0.0,e_center,f64::EPSILON, 
        e_center ).unwrap();
    println!("rho_center = {}, rho0_center_est = {}", rho_center, rho0_center_est);
    assert_approx_eq!(rho_center, rho0_center_est, 0.0001);
}

pub fn rtsec_g( func: &dyn Fn(f64, f64)-> f64, 
            gamma_p:  f64, 
            x1: f64, 
            x2: f64, 
            xacc: f64, 
            ee: f64) -> Result<f64,Box<dyn Error>> {


let mut rts:f64;
    let mut x_l:f64;
    let swap:f64;

    let mut f_l = func(x1,gamma_p)-ee;
    let mut f = func(x2,gamma_p)-ee;

    if f_l.abs() < f.abs() {
        rts=x1;
        x_l=x2;
        swap=f_l;
        f_l=f;
        f=swap;
    } else {
        x_l=x1;
        rts=x2;
    }

    for _j in 0..MAXIT {
    let dx=(x_l-rts)*f/(f-f_l);
    x_l=rts;
    f_l=f;
    rts += dx;
    f = func(rts,gamma_p)-ee;

    if dx.abs()<xacc || f==0.0 {
        return Ok(rts);
    }
    }

    Err("Maximum number of iterations exceeded in rtsec".into())  
}

fn deriv_s(f: &Array2<f64>, s: usize, m: usize) ->  f64 {

    match s {
        0 => (f[[s+1,m]]-f[[s,m]])/DS, // edge case
        1 ..= SDIV_MIN_2 => (f[[s+1,m]]-f[[s-1,m]])/(2.0*DS), // normal case
        SDIV_MIN_1  => (f[[s,m]]-f[[s-1,m]])/DS, // edge case        
        _ => panic!("Attempted to take derivative with index out of bounds in deriv_s."),
    }
}


fn deriv_m(f: &Array2<f64>, s: usize, m: usize) ->  f64 {

    match m {
        0 => (f[[s,m+1]]-f[[s,m]])/DM, // edge case
        1 ..= MDIV_MIN_2 => (f[[s,m+1]]-f[[s,m-1]])/(2.0*DM), // normal case
        MDIV_MIN_1  => (f[[s,m]]-f[[s,m-1]])/DM, // edge case        
        _ => panic!("Attempted to take derivative with index out of bounds in deriv_m."),
    }
}

fn deriv_sm(f: &Array2<f64>, s: usize, m: usize) ->  f64 {

    match m {
        0 => (deriv_s(f, s, m+1) - deriv_s(f, s, m))/DM, // edge case
        1 ..= MDIV_MIN_2 => (deriv_s(f, s, m+1) - deriv_s(f, s, m-1))/(2.0*DM), // normal case
        MDIV_MIN_1  => (deriv_s(f, s, m) - deriv_s(f, s, m-1))/DM, // edge case        
        _ => panic!("Attempted to take derivative with index out of bounds in deriv_sm."),
    }
}

#[test]
fn test_dr_dr_is(){
    let m= 0.32; // Not real numbers here, just testing function
    let r = 1.0;
    let r_is = 0.5;
    assert_approx_eq!(dr_dr_is(r_is, r, m), 1.2);
    let m= 0.32; // Not real numbers here, just testing function
    let r = 1.0;
    let r_is = RMIN * 0.5;
    assert_approx_eq!(dr_dr_is(r_is, r, m), 1.0);
}

#[test]
#[should_panic]
fn test_dr_dr_is_fail() {
    let m= 0.32; // Not real numbers here, just testing function
    let r = 0.32;
    let r_is = 0.32;
    dr_dr_is(r_is, r, m);
}

fn dr_dr_is(r_is: f64, r: f64, m: f64) -> f64 {

    if m/r > 0.5 {
        panic!("Collapsed to a black hole!!! aka m/r > 0.5 in dr_dr_is");
    } else if r_is < RMIN {
        1.0
    } else {
        r/r_is * (1.0-2.0 * m/r).sqrt()
    }
}

#[test]
fn test_dm_dr_is(){
    todo!();
}

fn dm_dr_is(r_is: f64, 
            r: f64, 
            m: f64, 
            p: f64, 
            e_center: f64, 
            p_surface: f64, 
            opt_log_e_tab: Option<Vec<f64>>, 
            opt_log_p_tab: Option<Vec<f64>>, 
            opt_nearest: Option<usize>, 
            eos_type: EosType, 
            opt_gamma_p: Option<f64>) -> f64 {

    let e_d  = if p<p_surface {
        0.0_f64 
        } else {
            e_at_p(p, 
                    opt_log_e_tab, opt_log_p_tab, eos_type, opt_gamma_p, opt_nearest).unwrap()
                };
    match r_is < RMIN {
        true => 4.0 * PI * e_center * r * r * (1.0 + 4.0 * PI * e_center * r * r / 3.0),
        false => 4.0 * PI * e_d * r * r * r * (1.0 - 2.0 * m / r).sqrt() / r_is,
    }
}



fn dp_dr_is(r_is: f64, 
    r: f64, 
    m: f64, 
    p: f64, 
    e_center: f64, 
    p_surface: f64, 
    opt_log_e_tab: Option<Vec<f64>>, 
    opt_log_p_tab: Option<Vec<f64>>, 
    opt_nearest: Option<usize>, 
    eos_type: EosType, 
    opt_gamma_p: Option<f64>) -> f64 {

    let e_d  = if p<p_surface {
    0.0_f64 
    } else {
        e_at_p(p, 
                opt_log_e_tab, opt_log_p_tab, eos_type, opt_gamma_p, opt_nearest).unwrap()
            };
    match r_is < RMIN {
        true => -4.0 * PI * (e_center + p) * (e_center + 3.0 * p) * r * (1.0 + 4.0 * e_center * r * r / 3.0) / 3.0,
        false => -(e_d + p) * (m + 4.0 * PI * r * r * r * p) / (r * r_is * (1.0-2.0*m/r).sqrt()),
    }
}


#[test]
fn test_tov() {
    todo!();
}

fn tov(i_check:  usize, 
        e_center: f64,
        eos_type: EosType, 
        opt_gamma_p: Option<f64>, 
        mut r_is_final: f64,) {

    let r_is_est: f64;
    let h: f64;
    let dr_is_save: f64;
    let r_is_save: f64;
    let r_is_check: f64;

    
    if i_check == 1 {
        r_is_est = match eos_type {
            EosType::Table => 1.5e6 / KAPPA.sqrt(),
            EosType::Polytropic => { 
                let gamma_p = opt_gamma_p.unwrap();
                2.0*(gamma_p/(4.0*PI*(gamma_p-1.0))).sqrt()*e_center.powf((gamma_p-2.0)/2.0)
            },
        };
        h = r_is_est / 100.0;
    } else {
        r_is_est = r_is_final;
        h = r_is_est / 10_000.0;
        dr_is_save = r_is_final / (RDIV as f64);
        r_is_check = dr_is_save;
    }


}



#[test]
fn test_sphere() {
    let (s,m) = make_grid(None,None);
    sphere(&s);

}

fn sphere(s_gp: &[f64]) {

    println!("{}", s_gp[20]);
    todo!();

}




// fn mass_radius(
//     s_gp: &[f64],
//     gama: Array2<f64>,
//     r_ratio : f64,
//     r_e: f64,
// ) {
//     let r_p = r_ratio * r_e;
//     let s_p = r_p/(r_p+r_e);
//     let s_e = 0.5;
//     let rho_0:Array2<f64> = Array2::zeros((SDIV, MDIV));
//     let velocity:Array2<f64> = Array2::zeros((SDIV, MDIV));

//     let gama_mu_0 = gamma
//     // let gama_mu_0:Vec<f64> = vec![0.0;SDIV];   

//     let gama_equator = interp(s_gp, gama, s_e, None);
    
// }