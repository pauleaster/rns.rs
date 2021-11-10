use std::cmp::{min,max};
use std::convert::TryInto;
use std::f64::consts::PI;
// use approx::{relative_eq, relative_ne};
use assert_approx_eq::assert_approx_eq;
use ndarray::{Array2, Array3, Zip, array, prelude::*};
use std::error::Error;
use std::time::Instant;


use crate::consts::*;
use crate::equil::{e_at_p, e_of_rho0, get_e_p_surface, get_min_enthalpy, h_at_p, load_eos, make_center, make_grid, p_at_e, p_at_h, read_eos_file};


pub enum EosType {
    Table,
    Polytropic,
}

#[test]
fn test_hunt() {
    
    let x1:Vec<f64> = (0 ..= 19).map(f64::from).collect();
    let y1:Vec<f64> = (0 ..= 19).map(|x| (x as f64).powi(2)).collect();
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
    let y1:Vec<f64> = (0 ..= 19).map(|x| 400. - (x as f64).powi(2)).collect();
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
        let (yb, _) = interp(xp, yp, xbval, Some(6));
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
            opt_nearest: Option<usize>) -> (f64, Option<usize>) { 

    let nearest = hunt(xp,xb, opt_nearest);

    let np = xp.len();
    let k=max(0,
                    min(nearest as i32 - 1, np as i32 - 4)) as usize;

    // epsilon shift corrected, should eliminate (xp[i]-xp[j]).abs() < eps


    let d0 = round_from_zero((xp[k]-xp[k+1])*(xp[k]-xp[k+2])*(xp[k]-xp[k+3]));
    let d1 = round_from_zero((xp[k+1]-xp[k])*(xp[k+1]-xp[k+2])*(xp[k+1]-xp[k+3]));
    let d2 = round_from_zero((xp[k+2]-xp[k])*(xp[k+2]-xp[k+1])*(xp[k+2]-xp[k+3]));
    let d3 = round_from_zero((xp[k+3]-xp[k])*(xp[k+3]-xp[k+1])*(xp[k+3]-xp[k+2]));


    ((xb-xp[k+1])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k]/ d0
        + (xb-xp[k])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k+1]/ d1 
        + (xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+3])*yp[k+2]/ d2
        + (xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+2])*yp[k+3]/ d3, Some(nearest))

    
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

    for _j in 0 ..= MAXIT {
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

// #[test]
// fn test_dm_dr_is(){
//     todo!();
// }

#[allow(clippy::too_many_arguments)]
fn dm_dr_is(r_is: f64, 
            r: f64, 
            m: f64, 
            p: f64, 
            e_center: f64, 
            p_surface: f64, 
            opt_log_e_tab: &Option<Vec<f64>>, 
            opt_log_p_tab: &Option<Vec<f64>>, 
            opt_nearest: Option<usize>, 
            eos_type: &EosType, 
            opt_gamma_p: Option<f64>) -> (f64, Option<usize>) {


    let (e_d, opt_nearest)  = if p<p_surface {
        (0.0_f64,Some(0))
        } else {
            e_at_p(p, 
                    opt_log_e_tab, opt_log_p_tab, eos_type, opt_gamma_p, opt_nearest).unwrap()
                };
    match r_is < RMIN {
        true => (4.0 * PI * e_center * r * r * (1.0 + 4.0 * PI * e_center * r * r / 3.0), opt_nearest),
        false => (4.0 * PI * e_d * r * r * r * (1.0 - 2.0 * m / r).sqrt() / r_is, opt_nearest),
    }
}


#[allow(clippy::too_many_arguments)]
fn dp_dr_is(r_is: f64, 
    r: f64, 
    m: f64, 
    p: f64, 
    e_center: f64, 
    p_surface: f64, 
    opt_log_e_tab: &Option<Vec<f64>>, 
    opt_log_p_tab: &Option<Vec<f64>>, 
    opt_nearest: Option<usize>, 
    eos_type: &EosType, 
    opt_gamma_p: Option<f64>) -> (f64, Option<usize>) {

    let (e_d, opt_nearest)  = if p<p_surface {
        (0.0_f64,Some(0))
        } else {
            e_at_p(p, 
                    opt_log_e_tab, opt_log_p_tab, eos_type, opt_gamma_p, opt_nearest).unwrap()
                };
        match r_is < RMIN {
            true => (-4.0 * PI * (e_center + p) * (e_center + 3.0 * p) * r * (1.0 + 4.0 * e_center * r * r / 3.0) / 3.0, opt_nearest),
            false => (-(e_d + p) * (m + 4.0 * PI * r * r * r * p) / (r * r_is * (1.0-2.0*m/r).sqrt()), opt_nearest),
        }
}


// #[test]
// fn test_tov() {
//     todo!();
// }

// #[derive(PartialEq)]
enum ICheck {
    Initial,
    Intermediate,
    Final,
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
fn tov(i_check:  ICheck,
        e_center: f64,
        p_center: f64,
        e_surface: f64,
        p_surface: f64,
        opt_log_e_tab: &Option<Vec<f64>>,
        opt_log_p_tab: &Option<Vec<f64>>,
        opt_log_h_tab: &Option<Vec<f64>>,
        eos_type: &EosType, 
        opt_gamma_p: Option<f64>, 
        r_is_gp : &mut [f64], 
        lambda_gp: &mut [f64], 
        nu_gp: &mut [f64],
        r_is_final: &mut f64, 
        r_final: &mut f64,
        m_final: &mut f64) {

    
    let mut i:usize = 1;
    // let mut nearest: usize;
    let mut opt_nearest: Option<usize> = None;

    
    let r_is_est: f64;
    
    let h: f64;
    let dr_is_save: f64;
    let mut r_is_check: f64;

    let mut r_gp = [0.; RDIV+1]; // Vec::with_capacity(RDIV+1);
    let mut m_gp = [0.; RDIV+1]; //  Vec::with_capacity(RDIV+1);
    let mut e_d_gp = [0.; RDIV+1]; //  Vec::with_capacity(RDIV+1);

    match i_check {
        ICheck::Initial => {
            r_is_est = match eos_type {
                EosType::Table => 1.5e6 / KAPPA.sqrt(),
                EosType::Polytropic => { 
                    let gamma_p = opt_gamma_p.unwrap();
                    2.0*(gamma_p/(4.0*PI*(gamma_p-1.0))).sqrt()*e_center.powf((gamma_p-2.0)/2.0)
                },
            };
            h = r_is_est / 100.0;
            r_is_check = 0.0; // Dummy variable to keep the compiler happy
            dr_is_save = 0.0; // Dummy variable to keep the compiler happy
        },
        ICheck::Intermediate | ICheck::Final => {
            r_is_est = *r_is_final;
            h = r_is_est / 10_000.0;
            dr_is_save = *r_is_final / (RDIV as f64);
            r_is_check = dr_is_save;
        }
    }

    let mut r_is=0.0;                            /* initial isotropic radius */
    let mut r=0.0;                               /* initial radius */
    let mut m=0.0;                               /* initial mass */ 
    let mut p=p_center;                          /* initial pressure */ 

    r_is_gp[0]=0.0;
    r_gp[0]=0.0;
    m_gp[0]=0.0;
    lambda_gp[0]=0.0;
    e_d_gp[0] = e_center; 

    while p>= p_surface {

        let (e_d, opt_nearest) = e_at_p(p,opt_log_e_tab,opt_log_p_tab,eos_type,opt_gamma_p,opt_nearest).unwrap();

        if matches!(i_check, ICheck::Final) && (r_is>r_is_check) && (i< RDIV) {
                    r_is_gp[i]=r_is;
                    r_gp[i]=r;
                    m_gp[i]=m;
                    e_d_gp[i]=e_d; 
                    i += 1;   
                    r_is_check += dr_is_save;
        }
        *r_is_final=r_is;
        *r_final=r;
        *m_final=m;

        let a1=dr_dr_is(r_is,r,m);

        let (b1, opt_nearest)=dm_dr_is(r_is,r,m,p, e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                                opt_nearest, eos_type, opt_gamma_p);
        let (c1, _) = dp_dr_is(r_is,r,m,p, e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
            opt_nearest, eos_type, opt_gamma_p);



        let a2=dr_dr_is(r_is+h/2.0, r+h*a1/2.0, m+h*b1/2.0);

        let (b2, opt_nearest)=dm_dr_is(r_is+h/2.0, r+h*a1/2.0, m+h*b1/2.0, p+h*c1/2.0, 
                                        e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                        opt_nearest, eos_type, opt_gamma_p);

        let (c2, _) = dp_dr_is(r_is+h/2.0, r+h*a1/2.0, m+h*b1/2.0, p+h*c1/2.0, 
                                                e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                                opt_nearest, eos_type, opt_gamma_p);



        let a3=dr_dr_is(r_is+h/2.0, r+h*a2/2.0, m+h*b2/2.0);

        let (b3, opt_nearest)=dm_dr_is(r_is+h/2.0, r+h*a2/2.0, m+h*b2/2.0, p+h*c2/2.0, 
                                                        e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                                        opt_nearest, eos_type, opt_gamma_p);

        let (c3, _)=dp_dr_is(r_is+h/2.0, r+h*a2/2.0, m+h*b2/2.0, p+h*c2/2.0, 
                                                e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                                opt_nearest, eos_type, opt_gamma_p);



        let a4=dr_dr_is(r_is+h, r+h*a3, m+h*b3);

        let (b4, opt_nearest)=dm_dr_is(r_is+h, r+h*a3, m+h*b3, p+h*c3, 
                                                    e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                                    opt_nearest, eos_type, opt_gamma_p);
        
        let (c4, _)=dp_dr_is(r_is+h, r+h*a3, m+h*b3, p+h*c3, 
                                        e_center, p_surface, opt_log_e_tab, opt_log_p_tab, 
                                        opt_nearest, eos_type, opt_gamma_p);

        r += (h/6.0)*(a1+2.*a2+2.*a3+a4);
        m += (h/6.0)*(b1+2.*b2+2.*b3+b4);
        p += (h/6.0)*(c1+2.*c2+2.*c3+c4);

        // println!("r,m,p = {:9.8e}, {:9.8e}, {:9.8e}",r,m,p);
    
        r_is += h;

    }
    r_is_gp[RDIV-1]= *r_is_final;
    r_gp[RDIV-1]= *r_final;
    m_gp[RDIV-1]= *m_final;

    /* Rescale r_is and compute lambda */

    if matches!(i_check, ICheck::Final) {
        let k_rescale=0.5*( (*r_final)/ (*r_is_final))*(1.0-(*m_final)/(*r_final)+
                                (1.0-2.0*(*m_final)/(*r_final)).sqrt());

        *r_is_final *= k_rescale;

        let nu_s = ((1.0 - (*m_final) / (2.0 * (*r_is_final)) ) /
                        (1.0 + (*m_final) / (2.0 * (*r_is_final)) ) ).ln();

        for i in 0..= RDIV-1 {
            r_is_gp[i] *= k_rescale;
            lambda_gp[i] = match i {
                0 => (1.0/k_rescale).ln(),
                _ => (r_gp[i]/r_is_gp[i]).ln(),
            };

            let hh = match e_d_gp[i]<e_surface {
                true => 0.0,
                false => match eos_type {
                    EosType::Table => {
                        let (p, _) = p_at_e(e_d_gp[i], opt_log_e_tab.as_ref().unwrap(), opt_log_p_tab.as_ref().unwrap(), opt_nearest);
                        h_at_p(p, opt_log_h_tab.as_ref().unwrap(), opt_log_p_tab.as_ref().unwrap(), opt_nearest).0
                    }, // Table
                    EosType::Polytropic => {
                        let rho0=rtsec_g(&e_of_rho0,opt_gamma_p.unwrap(),0.0,e_d_gp[i],f64::EPSILON,
                            e_d_gp[i]).unwrap();
                        let p=rho0.powf(opt_gamma_p.unwrap());
                        ((e_d_gp[i]+p)/rho0).ln()
                    } // Polytropic
                } // eos_type,
            }; // let


            nu_gp[i]=nu_s-hh;
        } // for i
        nu_gp[RDIV - 1]=nu_s;
    }
}




#[test]
fn test_sphere() {
    let (s,m) = make_grid();
    let (log_e_tab, log_p_tab, log_h_tab, _, _) = load_eos("./eos/eosA").unwrap(); 
    let eos_type = &EosType::Table;
    // let opt_gamma_p = None;
    let opt_log_e_tab = &Some(log_e_tab);
    let opt_log_p_tab = &Some(log_p_tab);
    let opt_log_h_tab = &Some(log_h_tab);
    let unscaled_e_center = 1e15;
    let e_center = unscaled_e_center * CC * CC * KSCALE;
    
    let (p_center, _) = make_center(opt_log_e_tab, 
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
    let mut r_e = 0.0;

    sphere(&s, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, None, 
            e_center, p_center, p_surface, e_surface, &mut rho, &mut gama, &mut alpha, &mut omega, &mut r_e);

    // Values calibrated from the RNS c code

    println!("e_center = {}",e_center);
    println!("rho[0][0] = {:13.9e}",rho[[0,0]]);
    assert_approx_eq!(rho[[0,0]], -5.479849684e-01);
    assert_approx_eq!(rho[[(SDIV>>1)-1,(MDIV>>1)-1]], -2.785717738e-01);
    assert_approx_eq!(rho[[SDIV-1,MDIV-1]], -2.743658910e-05);
    
    assert_approx_eq!(gama[[0,0]], -2.524611504e-02);
    assert_approx_eq!(gama[[(SDIV>>1)-1,(MDIV>>1)-1]],  -5.037446991e-03);
    assert_approx_eq!(gama[[SDIV-1,MDIV-1]], -4.704792253e-11);
    
    assert_approx_eq!(alpha[[0,0]], 2.613694267e-01);
    assert_approx_eq!(alpha[[(SDIV>>1)-1,(MDIV>>1)-1]], 1.367671634e-01);
    assert_approx_eq!(alpha[[SDIV-1,MDIV-1]], 1.371827103e-05);
    
    assert_approx_eq!(omega[[0,0]], 0.0);
    assert_approx_eq!(omega[[(SDIV>>1)-1,(MDIV>>1)-1]], 0.0);
    assert_approx_eq!(omega[[SDIV-1,MDIV-1]], 0.0);

    assert_approx_eq!(r_e, 2.373788814e-01);

}
#[allow(clippy::too_many_arguments)]
fn sphere(s_gp: &[f64;SDIV],
            opt_log_e_tab: &Option<Vec<f64>>,
            opt_log_p_tab: &Option<Vec<f64>>,
            opt_log_h_tab: &Option<Vec<f64>>,
            eos_type: &EosType, 
            opt_gamma_p: Option<f64>, 
            e_center: f64,
            p_center: f64, 
            // double h_center,
            p_surface: f64,
            e_surface: f64,
            rho: &mut Array2<f64>, // metric parameter
            gama: &mut Array2<f64>, // metric parameter
            alpha: &mut Array2<f64>, // metric parameter
            omega: &mut Array2<f64>, // metric parameter
            r_e: &mut f64) {
 
    let s_e = 0.5;

    let r_is_final = &mut 0.0;
    let r_final = &mut 0.0;
    let m_final = &mut 0.0;

    let r_is_gp= &mut[0_f64;RDIV];
    let lambda_gp = &mut[0_f64;RDIV];
    let nu_gp = &mut[0_f64;RDIV];
    let gama_mu_0 = &mut[0_f64;SDIV];
    let rho_mu_0 = &mut[0_f64;SDIV];

    /* The function TOV integrates the TOV equations. The function
    can be found in the file equil.c */

    tov(ICheck::Initial, e_center, p_center, e_surface, p_surface, opt_log_e_tab, 
        opt_log_p_tab, opt_log_h_tab, eos_type, opt_gamma_p,
        r_is_gp, lambda_gp, nu_gp, r_is_final, r_final, m_final);
        
    // println!("tov initial\n");
    // println!("*********************");
    // println!("r_is_gp=\n{:?}", r_is_gp);
    // println!("lambda_gp=\n{:?}", lambda_gp);
    // println!("nu_gp=\n{:?}", nu_gp);
    // println!("\nr_is_final, r_final, m_final = ({}, {}, {})\n",r_is_final, r_final, m_final);
    
    tov(ICheck::Intermediate, e_center, p_center, e_surface, p_surface, opt_log_e_tab, 
        opt_log_p_tab, opt_log_h_tab, eos_type, opt_gamma_p,
        r_is_gp, lambda_gp, nu_gp, r_is_final, r_final, m_final);

    // println!("tov Intermediate\n");
    // println!("*********************");
    // println!("r_is_gp=\n{:?}", r_is_gp);
    // println!("lambda_gp=\n{:?}", lambda_gp);
    // println!("nu_gp=\n{:?}", nu_gp);
    // println!("\nr_is_final, r_final, m_final = ({}, {}, {})\n",r_is_final, r_final, m_final);

    tov(ICheck::Final, e_center, p_center, e_surface, p_surface, opt_log_e_tab, 
        opt_log_p_tab, opt_log_h_tab, eos_type, opt_gamma_p,
        r_is_gp, lambda_gp, nu_gp, r_is_final, r_final, m_final);

    // println!("tov Final\n");
    // println!("*********************");
    // println!("r_is_gp=\n{:?}", r_is_gp);
    // println!("lambda_gp=\n{:?}", lambda_gp);
    // println!("nu_gp=\n{:?}", nu_gp);
    println!("\nr_is_final, r_final, m_final = ({}, {}, {})\n",r_is_final, r_final, m_final);


    let nearest= RDIV >> 1;
    for s in 0..= SDIV-1  {
        let r_is_s=*r_is_final*(s_gp[s]/(1.0-s_gp[s]));

        let (nu_s, lambda_s) = if r_is_s < *r_is_final {
            let (lambda_s, opt_nearest)=interp(r_is_gp,lambda_gp,r_is_s,Some(nearest));
            let (nu_s, _)=interp(r_is_gp,nu_gp,r_is_s, opt_nearest);
            (nu_s, lambda_s)
            }
            else {
                let lambda_s = 2.0 * (1.0 + *m_final / (2.0 * r_is_s )).ln();
                let nu_s=((1.0 - *m_final / (2.0 * r_is_s) )/(1.0 + *m_final/(2.0 * r_is_s))).ln();
                (nu_s, lambda_s)
            };

        gama[[s,0]]=nu_s+lambda_s;
        rho[[s,0]]=nu_s-lambda_s;

        for m in 0 ..= MDIV-1 {
            gama[[s,m]]=gama[[s,0]];        
            rho[[s,m]]=rho[[s,0]];
            alpha[[s,m]]=(gama[[s,0]]-rho[[s,0]])/2.0;
            omega[[s,m]]=0.0; 
        }

        gama_mu_0[s]=gama[[s,0]];                   /* gama at \mu=0 */
        rho_mu_0[s]=rho[[s,0]];                     /* rho at \mu=0 */

    }

    let opt_nearest = Some(SDIV>>1);
    let (gama_eq, opt_nearest) = interp(s_gp,gama_mu_0,s_e,opt_nearest); /* gama at equator */
    let (rho_eq, _) = interp(s_gp,rho_mu_0,s_e, opt_nearest);   /* rho at equator */

    (*r_e)= *r_final * (0.5*(rho_eq-gama_eq)).exp(); 

}


#[test]
fn test_legendre() {
    let x = [0.2, 0.5, 0.8];
    let order = [2,6,9];
    let expected_results = [-0.44, 0.32324, 0.18786];

    for (i,&xval) in x.iter().enumerate(){
        assert_approx_eq!(legendre(order[i], xval),expected_results[i], 0.00001);
    }

}
/*******************************************************************/
/* Returns the Legendre polynomial of degree n, evaluated at x.    */
/*******************************************************************/
fn legendre( n: usize, x: f64 ) -> f64 {

    // p Legendre polynomial of order n 
    // p_1   "      "      "    "   n-1
    // p_2   "      "      "    "   n-2 

    match n {
        0 => 1.0,
        1 => x,
        2.. => {
            let mut p = 0.0;
            let mut p_2=1.0;
            let mut p_1=x;
            for i in 2..=n {
                p = (x*(2.0 * (i as f64) - 1.0) * p_1 - ((i as f64) - 1.0) * p_2) / (i as f64);
                p_2 = p_1;
                p_1 = p;
            }
            p
        },
        _ => 0.0, // never executed
    }

}


#[test]
fn test_legendre_lm() {
    let lvals = [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8];
    let mvals = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5];
    let results = [1.0, 0.5, -0.125, -0.4375, -0.8660254037844386, -1.299038105676658, 
                            -0.3247595264191645, 1.3531646934131853, 2.25, 5.625, 4.21875, -4.921875, 
                            -9.742785792574933, -34.099750274012266, -42.62468784251533, 12.787406352754601, 
                            59.06249999999999, 265.78124999999994, 465.11718749999994, 121.81640625, 
                            -460.3466286991656, -2531.906457845411, -5696.789530152175, -4114.347993998793];
    for i in 0 ..= lvals.len()-1 {
        assert_approx_eq!(legendre_poly_lm(lvals[i], mvals[i], 0.5), results[i]);
    } 

}

/*******************************************************************/
/* Returns the associated Legendre polynomial P_l^m(x).            */
/* Adapted from numerical recipes.                                 */
/*******************************************************************/
fn legendre_poly_lm(l: i32, m: i32, x: f64) -> f64 { // renamed from plgndr()
	// double fact,pll,pmm,pmmp1,somx2;
	// int i,ll;

    let mut pll = 0.0;
    let mut pmmp1;
    let mut fact;

	if  x.abs() > 1.0 {
		panic!("|x| must be <= 1.0 in legendre_poly_lm (plgndr), x={}",x);
    }
    if m < 0 {
        panic!("Negative m value passed to legendre_poly_lm Poly (plgndr), m={}",m);
    }
	let mut pmm=1.0;

    let somx2=((1.0-x)*(1.0+x)).sqrt();
    fact=1.0;
    for _ in 1 ..= m { //(i=1;i<=m;i++) {
        pmm *= -fact*somx2;
        fact += 2.0;
    }

    match l-m {
        i32::MIN..=-1 => panic!("m>l is invalid in Legendre_lm Poly (plgndr), m={} and l={}",m,l),
        0 => pmm,
        1 =>  x * (2.0 * m as f64 + 1.0 ) * pmm,
        2 .. => {
            pmmp1=x * (2.0 * m as f64 + 1.0 ) * pmm;
            for ll in m+2 ..= l { //(ll=(m+2);ll<=l;ll++) {
				pll=(x * (2.0 * ll as f64 - 1.0) * pmmp1 - ((ll+m-1) as f64) * pmm) / ((ll-m) as f64);
				pmm=pmmp1;
				pmmp1=pll;
			}
			pll
        },
    }

}


#[test]
fn test_spin() {
    let (s,m) = make_grid();
    let (log_e_tab, log_p_tab, log_h_tab, _, _) = load_eos("./eos/eosA").unwrap(); 
    let eos_type = &EosType::Table;
    // let opt_gamma_p = None;
    let opt_log_e_tab = &Some(log_e_tab);
    let opt_log_p_tab = &Some(log_p_tab);
    let opt_log_h_tab = &Some(log_h_tab);
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

    let r_ratio = 1.0;
    let enthalpy_min = get_min_enthalpy( eos_type);
    let a_check = &mut 0;
    let accuracy =1e-5;
    let cf = 1.0;
    let r_e_new = &mut 0.0;
    let big_omega = &mut 0.0;
    

    spin(&s, &m, opt_log_e_tab, opt_log_p_tab, opt_log_h_tab, eos_type, &None, h_center,
        enthalpy_min, rho, gama, alpha, omega, energy, pressure, enthalpy, velocity_sq,
        a_check, accuracy, cf, r_ratio, r_e_new, big_omega);
    println!("Finished");


}

#[allow(clippy::too_many_arguments)]
/*************************************************************************/
/* Main iteration cycle for computation of the rotating star's metric    */
/*************************************************************************/
fn spin(
    s_gp : &[f64],
    mu: &[f64],
    opt_log_e_tab: &Option<Vec<f64>>, 
    opt_log_p_tab: &Option<Vec<f64>>, 
    opt_log_h_tab: &Option<Vec<f64>>, 
    eos_type: &EosType,
    opt_gamma_p: &Option <f64>, 
    h_center: f64,
    enthalpy_min: f64,
    rho: &mut Array2<f64>,
    gama: &mut Array2<f64>,
    alpha: &mut Array2<f64>,
    omega: &mut Array2<f64>,
    energy: &mut Array2<f64>,
    pressure: &mut Array2<f64>,
    enthalpy: &mut Array2<f64>,
    velocity_sq: &mut Array2<f64>,
    a_check: &mut i32, 
    accuracy: f64,
    cf: f64,
    r_ratio: f64,
    r_e_new: &mut f64,
    big_omega: &mut f64
)

{

    let mut n_of_it=0;              /* number of iterations */
    let print_dif = true;
    let mut dif=1.0;             /* difference | r_e_old - r_e | */
    let mut term_in_big_omega_h;
    let mut omega_equator_h;             /* omega^hat at equator */         
    let s_e=0.5;
    let mut big_omega_h = 0.0;
    
    let f_rho = &mut Array3::<f64>::zeros((SDIV, LMAX + 1, SDIV)); //f3tensor(1,SDIV,1,LMAX+1,1,SDIV);
    let f_gama = &mut Array3::<f64>::zeros((SDIV, LMAX + 1, SDIV)); //f3tensor(1,SDIV,1,LMAX+1,1,SDIV);

    let p_2n = &mut Array2::<f64>::zeros((MDIV,LMAX + 1));   // dmatrix(1,MDIV,1,LMAX+1); 
    let p1_2n_1 = &mut Array2::<f64>::zeros((MDIV,LMAX + 1)); //dmatrix(1,MDIV,1,LMAX+1); 

    { // scope to allow the freeing of f2n
        let f2n = &mut Array2::<f64>::zeros((LMAX + 1, SDIV)); //dmatrix(1,LMAX+1,1,SDIV);
        for n in 0 ..= LMAX {
            for i  in 1 ..= SDIV-1 {
                f2n[[n,i]] = ((1.0 - s_gp[i])/ s_gp[i]).powi(2 * n as i32);
                print!("f2n[[{},{}]] = {:0.12e}      ",n,i,f2n[[n,i]]);
                if n < LMAX {
                    println!();
                }
                else {
                    println!("s_gp[{}] = {:0.12e}",i,s_gp[i]);
                }

            }
        }
        let xxx: f64 = 9.920882812500e-1;
        let qqq:f64 = (1.0-xxx)/xxx;
        println!("(1-0.92)/0.92 = {:0.12e}", qqq);
        println!("qqq.powi(20) = {:0.12e}",qqq.powi(20_i32));
        // for (n, mut row) in f2n.axis_iter_mut(Axis(0)).enumerate() {
        //     for (i, val) in row.iter_mut().enumerate() {
        //         if i==0 {
        //             continue;
        //         }
        //         *val =  ((1.0 - s_gp[i])/ s_gp[i]).powi(2*(n+ 1) as i32); // here: n+1 = 1..=LMAX+1, C code: n+1: 1..=LMAX+1
        //     }
        // }
        

        // The published code has defined SMAX = 0.999, in this case the following code is always executed
        // The code corresponding to SMAX = 1.0 has been removed. It can be reinstated at a later date if required once the rust code is working
        // C code: if (SMAX != 1.0)  {  

        for j in 1..= SDIV-1 { // (j=2;j<=SDIV;j++)  *** j reduced by 1, j=0 not parsed
            for n in 0..= LMAX-1 { //(n=1;n<=LMAX;n++)    ***** n reduced by 1
                for k in 1 ..= SDIV-1 { //(k=2;k<=SDIV;k++) { ***** k reduced by 1, k=0 not parsed
                    let sk=s_gp[k];
                    let sj=s_gp[j];
                    let sk1=1.0-sk;
                    let sj1=1.0-sj;

                    if k<j  {   
                        f_rho[[j,n+1,k]] = f2n[[n+1,j]]*sj1/(sj*f2n[[n+1,k]]*sk1*sk1); // here: n+1 index = 1..=LMAX, C code: n+1 = 2..=LMAX+1
                        f_gama[[j,n+1,k]] = f2n[[n+1,j]]/(f2n[[n+1,k]]*sk*sk1); // here: j,k index = 1..=SDIV-1, C code: j,k = 2..=SDIV
                    } else {     
                        f_rho[[j,n+1,k]] = f2n[[n+1,k]]/(f2n[[n+1,j]]*sk*sk1);
                        f_gama[[j,n+1,k]] = f2n[[n+1,k]]*sj1*sj1*sk/(sj*sj*f2n[[n+1,j]]*sk1*sk1*sk1);
                    }
                } // for k
            } // for n
        } // for j

        for k in 1..= SDIV-1 { //(k=2;k<=SDIV;k++) {
            let sk=s_gp[k]; // here: k index = 1..=SDIV-1, C code: k index = 2..=SDIV
            f_rho[[0,0,k]]=1.0/(sk*(1.0-sk)); // here: f_rho[[0,0,k]], C code: f_rho[j=1][n+1=1][k]
        }


        for k in 1..= SDIV-1 { //(k=2;k<=SDIV;k++) {
            let sk=s_gp[k]; // here: k index = 1..=SDIV-1, C code: k index = 2..=SDIV
            let sk1=1.0-sk;         
            f_rho[[0,1,k]]=0.0;
            f_gama[[0,1,k]]=1.0/(sk*sk1); // here: f_rho[[0,1,k]], f_gama[[0,1,k]], C code: f_rho[j = 1][n+1 = 2][k], f_gama[j][n+1][k]
        }

        for n in 1..= LMAX-1 { //(n=2;n<=LMAX;n++) {
            for k in 0..= SDIV-1 { //(k=1;k<=SDIV;k++) {
                f_rho[[0,n+1,k]]=0.0; // C code f_rho[j=1][n+1][k], f_gama[j=1][n+1][k]
                f_gama[[0,n+1,k]]=0.0;
            } 
        }

        for j in 0 ..= SDIV-1 { // for(j=1;j<=SDIV;j++)
            f_rho[[j,0,0]]=0.0; // C code f_rho[j][n+1 = 1][k = 1]
        }

        for j in 0 ..= SDIV-1 { // for(j=1;j<=SDIV;j++)
            for n in 0..= LMAX-1 {  // for(n=1;n<=LMAX;n++)
                f_rho[[j,n+1,0]]=0.0; // C code: f_rho[j][n+1][k=1], f_gama[j][n+1][k=1]
                f_gama[[j,n+1,0]]=0.0;
            }
        }


    for j in 1 ..= SDIV-1 { // for(j=2;j<=SDIV;j++)
        for k in 1 ..= SDIV-1 { // for(k=2;k<=SDIV;k++)
                let sk=s_gp[k];
                let sj=s_gp[j];
                let sk1=1.0-sk;
                let sj1=1.0-sj;

                if k<j {
                    f_rho[[j,0,k]] = sj1/(sj*sk1*sk1); // C code f_rho[j][n+1 = 1][k]
                }
                else {
                    f_rho[[j,0,k]] = 1.0/(sk*sk1);
                }
            }
        }
        // } C code if (SMAX != 1.0)


        // Note that there is no shift in the index of n in the legendre() function, ie. it is unchanged from the C code.
        for i in 0 ..= MDIV-1 {  //for(i=1;i<=MDIV;i++)
            p_2n[[i,0]]=legendre(0,mu[i]); // p_2n[i][n+1]=legendre(2*n,mu[i]);
        }
        
        for i in 0 ..= MDIV-1 { // for(i=1;i<=MDIV;i++)
            for n in 0 ..= LMAX - 1 { // for(n=1;n<=LMAX;n++) {
                p_2n[[i,n]]=legendre(2*n + 2,mu[i]); // 2(n+1) = 2n + 2
                p1_2n_1[[i,n]] = legendre_poly_lm((2*n +1) as i32 ,1,mu[i]); // 2(n+1) - 1 = 2n + 1
            }
        } 
    } // free_dmatrix(f2n,1,LMAX+1,1,SDIV);, f2n automatically freed here as it falls out of scope, 


    

    let sin_theta = &mut [0_f64;MDIV];
    let theta = &mut [0_f64;MDIV];
    
    for m in 0 ..= MDIV - 1 { //for(m=1;m<=MDIV;m++) { 
        sin_theta[m] = (1.0-mu[m]*mu[m]).sqrt();   // Why not acos(mu[m]) ?
        theta[m] = sin_theta[m].asin();
    }


    let mut r_e = *r_e_new;

    // main loop
    while dif> accuracy || n_of_it < 2 { 

        if print_dif  {
            println!("{:4.3e}",dif); 
        }



        /* Rescale potentials and construct arrays with the potentials along
        | the equatorial and polar directions.
        */
        
        let rho_mu_0= &mut [0.0_f64;SDIV];     
        let gama_mu_0= &mut [0.0_f64;SDIV];   
        let omega_mu_0= &mut [0.0_f64;SDIV];  
        let rho_mu_1= &mut [0.0_f64;SDIV]; 
        let gama_mu_1= &mut [0.0_f64;SDIV]; 
        
        for s in 0 ..= SDIV-1 { //for(s=1;s<=SDIV;s++) {
            for m in 0 ..= MDIV-1 { //for(m=1;m<=MDIV;m++) {
                let re_sq = r_e.powi(2);
                rho[[s,m]] /= re_sq;
                gama[[s,m]] /= re_sq; 
                alpha[[s,m]] /= re_sq;
                omega[[s,m]] *= r_e; 
            }
            rho_mu_0[s]=rho[[s,0]];     
            gama_mu_0[s]=gama[[s,0]];   
            omega_mu_0[s]=omega[[s,0]]; 
            rho_mu_1[s]=rho[[s,MDIV-1]];  
            gama_mu_1[s]=gama[[s,MDIV-1]];
        }

        /* Compute new r_e. */ 

        let r_e_old= r_e;
        let r_p= r_ratio*r_e;                          
        let s_p=r_p/(r_p+r_e);                        

        let opt_nearest= Some(SDIV>>1);
        let (gama_pole_h, opt_nearest)=interp(s_gp,gama_mu_1,s_p,opt_nearest); 
        let (gama_equator_h, opt_nearest)=interp(s_gp,gama_mu_0,s_e,opt_nearest);
        let gama_center_h=gama[[0,0]];                    

        let (rho_pole_h,opt_nearest)=interp(s_gp,rho_mu_1,s_p,opt_nearest);   
        let (rho_equator_h,opt_nearest)=interp(s_gp,rho_mu_0,s_e,opt_nearest);
        let rho_center_h=rho[[0,0]];                      

        r_e=(2.0*h_center/(gama_pole_h+rho_pole_h-gama_center_h-rho_center_h)).sqrt();
        let re_sq = 2.0*h_center/(gama_pole_h+rho_pole_h-gama_center_h-rho_center_h);


        /* Compute angular velocity big_omega. */

        if approx::abs_diff_eq!(r_ratio, 1.0) {
            big_omega_h=0.0;
            omega_equator_h=0.0;
        } 
        else {
            omega_equator_h = interp(s_gp,omega_mu_0,s_e, opt_nearest).0;
            term_in_big_omega_h=1.0-(re_sq*(gama_pole_h+rho_pole_h-gama_equator_h-rho_equator_h)).exp();
            if term_in_big_omega_h>=0.0 {
                big_omega_h = omega_equator_h + (re_sq*rho_equator_h).exp()*(term_in_big_omega_h).sqrt();
            }
            else {
                big_omega_h=0.0;
            }
        }


        /* Compute velocity, energy density and pressure. */


        let opt_nearest=Some(opt_log_e_tab.as_ref().unwrap().len()>>1);

        for s in 0 ..= SDIV-1{ // (s=1;s<=SDIV;s++) {
            let sgp=s_gp[s];

            for m in  0 ..= MDIV-1 { //(m=1;m<=MDIV;m++) {
                let rsm=rho[[s,m]];

                if approx::abs_diff_eq!(r_ratio, 1.0, epsilon=f64::EPSILON)  {
                    velocity_sq[[s,m]]=0.0;
                }
                else  {
                    velocity_sq[[s,m]]=((big_omega_h-omega[[s,m]])*(sgp/(1.0-sgp))*sin_theta[m]*(-rsm*re_sq).exp()).powi(2);
                }

                if velocity_sq[[s,m]]>=1.0 {
                    velocity_sq[[s,m]]=0.0;
                }

                enthalpy[[s,m]]=enthalpy_min + 0.5*(re_sq*(gama_pole_h+rho_pole_h-gama[[s,m]]-rsm)-(1.0-velocity_sq[[s,m]]).ln());

                if (enthalpy[[s,m]]<=enthalpy_min) || (sgp>s_e) {
                    pressure[[s,m]]=0.0;
                    energy[[s,m]]=0.0; 
                }
                else { 
                    match eos_type {
                        EosType::Table => {
                            let p_h_n=p_at_h(enthalpy[[s,m]], &opt_log_p_tab.as_ref().unwrap(), &opt_log_h_tab.as_ref().unwrap(), opt_nearest);
                            pressure[[s,m]] = p_h_n.0;
                            let opt_nearest = p_h_n.1;
                            let e_p_n=e_at_p(pressure[[s,m]], opt_log_e_tab, opt_log_p_tab, &eos_type, None, opt_nearest).unwrap();
                            energy[[s,m]] = e_p_n.0;
                            let opt_nearest = e_p_n.1;
                        },
                        EosType::Polytropic => {
                            let gamma_p = opt_gamma_p.unwrap();
                            let rho0sm= (((gamma_p-1.0)/gamma_p)*((enthalpy[[s,m]]).exp()-1.0)).powf(1.0/(gamma_p-1.0));
                            pressure[[s,m]]=rho0sm.powf(gamma_p);
                            energy[[s,m]]=pressure[[s,m]]/(gamma_p-1.0)+rho0sm;
                        },
                    }
                } // else enthalpy



                /* Rescale back metric potentials (except omega) */

                rho[[s,m]] *= re_sq;
                gama[[s,m]] *= re_sq;
                alpha[[s,m]] *= re_sq;
            } // for m
        } // for s


        {

            /* Compute metric potentials */

            let  s_gama = &mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);
            let s_rho =&mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);
            let s_omega =&mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);

            for s in 0 ..= SDIV-1 { // (s=1;s<=SDIV;s++)
                for m in 0 ..= MDIV-1 { // (m=1;m<=MDIV;m++) {
                    let rsm=rho[[s,m]];
                    let gsm=gama[[s,m]];
                    let omsm=omega[[s,m]];
                    let esm=energy[[s,m]];
                    let psm=pressure[[s,m]];
                    let e_gsm=(0.5*gsm).exp();
                    let e_rsm=(-rsm).exp();
                    let v2sm=velocity_sq[[s,m]];
                    let mum=mu[m];            
                    let m1=1.0-mum.powi(2);
                    let sgp=s_gp[s];
                    let s_1=1.0-sgp;
                    let s1=sgp*s_1;
                    let s2=(sgp/s_1).powi(2);  

                    let ea=16.0*PI*(2.0*alpha[[s,m]]).exp()*re_sq;

                    // C code: if s==1 
                    let mut d_gama_s=0.0;
                    let mut d_gama_m=0.0;
                    let mut d_rho_s=0.0;
                    let mut d_rho_m=0.0;
                    let mut d_omega_s=0.0;
                    let mut d_omega_m=0.0;

                    if s > 0 { 
                        d_gama_s=deriv_s(gama,s,m);
                        d_gama_m=deriv_m(gama,s,m);
                        d_rho_s=deriv_s(rho,s,m);
                        d_rho_m=deriv_m(rho,s,m);
                        d_omega_s=deriv_s(omega,s,m);
                        d_omega_m=deriv_m(omega,s,m);
                    } // if s > 0

                    s_rho[[s,m]] = e_gsm * (0.5 * ea * (esm + psm) * s2 * (1.0+v2sm)/(1.0-v2sm) + s2 * m1 * e_rsm.powi(2) * ((s1 * d_omega_s).powi(2)
                                    + m1 * (d_omega_m).powi(2)) + s1 * d_gama_s - mum * d_gama_m + 0.5 * rsm * (ea * psm * s2 
                                    - s1 * d_gama_s * (0.5 * s1 * d_gama_s+1.0) - d_gama_m * (0.5 * m1 * d_gama_m-mum)));

                    s_gama[[s,m]] = e_gsm * (ea * psm * s2 + 0.5 * gsm * (ea * psm * s2 - 0.5 * (s1 * d_gama_s).powi(2) - 0.5 * m1 * (d_gama_m).powi(2)));

                    s_omega[[s,m]]=e_gsm * e_rsm * ( -ea * (big_omega_h-omsm) * (esm+psm) *s2/(1.0-v2sm) + omsm * ( -0.5 * ea * (((1.0+v2sm) * esm 
                                + 2.0 * v2sm * psm)/(1.0-v2sm)) * s2 - s1 * (2.0 * d_rho_s+0.5 * d_gama_s)
                                + mum * (2.0 * d_rho_m+0.5 * d_gama_m) + 0.25 * s1.powi(2) * (4.0 * d_rho_s.powi(2)- d_gama_s.powi(2) ) 
                                + 0.25 * m1 * (4.0 * d_rho_m.powi(2) - d_gama_m.powi(2) ) - m1 * e_rsm.powi(2) * ( (sgp.powi(2) * d_omega_s).powi(2) 
                                + s2 * m1 * d_omega_m.powi(2) )));
                } // for m
            } // for s



            /* ANGULAR INTEGRATION */

            let  d1_rho = &mut Array2::<f64>::zeros((LMAX+1,SDIV));  // dmatrix(1,LMAX+1,1,SDIV);
            let  d1_gama = &mut Array2::<f64>::zeros((LMAX+1,SDIV));  // dmatrix(1,LMAX+1,1,SDIV);
            let  d1_omega = &mut Array2::<f64>::zeros((LMAX+1,SDIV));  // dmatrix(1,LMAX+1,1,SDIV);

            let mut sum_rho = 0.0;
            let mut sum_gama = 0.0;
            let mut sum_omega = 0.0;

            for k in 0 ..= SDIV-1 { //for(k=1;k<=SDIV;k++) {      
                for m in (0 ..= MDIV-3).step_by(2) { //for(m=1;m<=MDIV-2;m+=2) {
                    sum_rho += (DM/3.0) * (p_2n[[m,0]] * s_rho[[k,m]]
                                + 4.0 * p_2n[[m+1,0]] * s_rho[[k,m+1]] 
                                + p_2n[[m+2,0]] * s_rho[[k,m+2]]);
                }

                d1_rho[[0,k]]=sum_rho;
                d1_gama[[0,k]]=0.0;
                d1_omega[[0,k]]=0.0;
                sum_rho=0.0;

            } // for k

            for n in 0 ..= LMAX-1 { // for(n=1;n<=LMAX;n++) {
                for k in 0 ..= SDIV-1 { //for(k=1;k<=SDIV;k++) {      
                    for m in 0 ..= MDIV-3 { // for(m=1;m<=MDIV-2;m+=2) {

                        // NOTE: any n values that are not indices must be replaced with (n+1) as f64

                        sum_rho += DM / 3.0 * (p_2n[[m,n]] * s_rho[[k,m]]
                                + 4.0 * p_2n[[m+1,n]] * s_rho[[k,m+1]] 
                                + p_2n[[m+2,n]] * s_rho[[k,m+2]]);
                                
                        sum_gama += DM / 3.0 * ( ((2.0 * (n+1) as f64 - 1.0) * theta[m]).sin() * s_gama[[k,m]]  // n => (n+1) as f64
                                    + 4.0 * ((2.0 * (n+1) as f64 - 1.0).sin() * theta[m+1]) * s_gama[[k,m+1]]
                                    + ((2.0 * (n+1) as f64 - 1.0) * theta[m+2]).sin() * s_gama[[k,m+2]]);

                        sum_omega += DM / 3.0 * (sin_theta[m] * p1_2n_1[[m,n+1]] * s_omega[[k,m]]
                                    +4.0 * sin_theta[m+1] * p1_2n_1[[m+1,n+1]] * s_omega[[k,m+1]]
                                    +sin_theta[m+2] * p1_2n_1[[m+2,n+1]] * s_omega[[k,m+2]]);
                    } // for m
                    d1_rho[[n+1,k]]=sum_rho;
                    d1_gama[[n+1,k]]=sum_gama;
                    d1_omega[[n+1,k]]=sum_omega;
                    sum_rho=0.0;
                    sum_gama=0.0;
                    sum_omega=0.0;
                } // for k
            } // for n

            // free_dmatrix(s_gama,1,SDIV,1,MDIV);
            // free_dmatrix(s_rho,1,SDIV,1,MDIV);
            // free_dmatrix(s_omega,1,SDIV,1,MDIV);
            


            /* RADIAL INTEGRATION */

            let  d2_rho = &mut Array2::<f64>::zeros((SDIV,LMAX+1));  // dmatrix(1,SDIV,1,LMAX+1);
            let  d2_gama = &mut Array2::<f64>::zeros((SDIV,LMAX+1));  // dmatrix(1,SDIV,1,LMAX+1);
            let  d2_omega = &mut Array2::<f64>::zeros((SDIV,LMAX+1));  // dmatrix(1,SDIV,1,LMAX+1);



            for s in 0 ..= SDIV-1 { //for(s=1;s<=SDIV;s++) {
                for k in (0 ..= SDIV-3).step_by(2) { // for(k=1;k<=SDIV-2;k+=2) { 
                    sum_rho += (DS/3.0) * ( f_rho[[s,0,k]] * d1_rho[[0,k]] 
                                + 4.0 * f_rho[[s,0,k+1]] * d1_rho[[0,k+1]]
                                + f_rho[[s,0,k+2]] *d1_rho[[0,k+2]]);
                    }
                d2_rho[[s,0]]=sum_rho;
                d2_gama[[s,0]]=0.0;
                d2_omega[[s,0]]=0.0;
                sum_rho=0.0;
            } // for s


            for s in 0 ..= SDIV-1 { //for(s=1;s<=SDIV;s++)
                for n in 0 ..= LMAX-1 { //for(n=1;n<=LMAX;n++) {
                    for k in (0 ..= SDIV-3).step_by(2) { // for(k=1;k<=SDIV-2;k+=2) { 
                        sum_rho += (DS/3.0) * (f_rho[[s,n+1,k]] * d1_rho[[n+1,k]] 
                                    + 4.0 * f_rho[[s,n+1,k+1]] * d1_rho[[n+1,k+1]]
                                    + f_rho[[s,n+1,k+2]] * d1_rho[[n+1,k+2]]);

                        sum_gama += (DS/3.0) * ( f_gama[[s,n+1,k]] * d1_gama[[n+1,k]] 
                                    + 4.0 * f_gama[[s,n+1,k+1]] * d1_gama[[n+1,k+1]]
                                    + f_gama[[s,n+1,k+2]] * d1_gama[[n+1,k+2]]);

                        if k<s && k+2<=s  {
                            sum_omega += (DS/3.0)*( f_rho[[s,n+1,k]]*d1_omega[[n+1,k]] 
                                            + 4.0*f_rho[[s,n+1,k+1]]*d1_omega[[n+1,k+1]]
                                            + f_rho[[s,n+1,k+2]]*d1_omega[[n+1,k+2]]);
                        }
                        else if k>=s {
                            sum_omega += (DS/3.0)*( f_gama[[s,n+1,k]]*d1_omega[[n+1,k]] 
                                        + 4.0*f_gama[[s,n+1,k+1]]*d1_omega[[n+1,k+1]]
                                        + f_gama[[s,n+1,k+2]]*d1_omega[[n+1,k+2]]);
                        }
                        else {
                            sum_omega += (DS/3.0)*( f_rho[[s,n+1,k]]*d1_omega[[n+1,k]] 
                                        + 4.0*f_rho[[s,n+1,k+1]]*d1_omega[[n+1,k+1]]
                                        + f_gama[[s,n+1,k+2]]*d1_omega[[n+1,k+2]]);
                        } // else not(k<s && k+2<=s)
                    } // for k
                    d2_rho[[s,n+1]]=sum_rho;
                    d2_gama[[s,n+1]]=sum_gama;
                    d2_omega[[s,n+1]]=sum_omega;
                    sum_rho=0.0;
                    sum_gama=0.0;
                    sum_omega=0.0;
                } // for n
            } // for s

            // free_dmatrix(d1_rho,1,LMAX+1,1,SDIV);
            // free_dmatrix(d1_gama,1,LMAX+1,1,SDIV);
            // free_dmatrix(d1_omega,1,LMAX+1,1,SDIV);


            /* SUMMATION OF COEFFICIENTS */

            for s in 0 ..= SDIV-1 { //for(s=1;s<=SDIV;s++) 
                for m in 0 ..= MDIV-1 { // for(m=1;m<=MDIV;m++) {

                    let gsm=gama[[s,m]];
                    let rsm=rho[[s,m]];
                    let omsm=omega[[s,m]];             
                    let e_gsm=(-0.5*gsm).exp();
                    let e_rsm= rsm.exp();
                    let temp1=sin_theta[m];

                    sum_rho += -e_gsm*p_2n[[m,0]]*d2_rho[[s,0]];  // reduce minimum n indices by 1, C code: p_2n[m][0+1]*d2_rho[[s,0+1]]

                    for n in 0 ..= LMAX-1 { // for(n=1;n<=LMAX;n++) {

                        sum_rho += -e_gsm*p_2n[[m,n]]*d2_rho[[s,n+1]]; 

                        if m==MDIV {             
                            sum_omega += 0.5*e_rsm*e_gsm*d2_omega[[s,n+1]]; 
                            sum_gama += -(2.0/PI)*e_gsm*d2_gama[[s,n+1]];   
                        }
                        else { 
                            sum_omega += -e_rsm*e_gsm*(p1_2n_1[[m,n+1]]/(2.0 * (n+1) as f64
                                            *(2.0 * (n+1) as f64 - 1.0)*temp1))*d2_omega[[s,n+1]];

                            sum_gama += -(2.0/PI)*e_gsm*( ((2.0 * (n+1) as f64-1.0)*theta[m]).sin()
                                        /((2.0 * (n+1) as f64 - 1.0)*temp1))*d2_gama[[s,n+1]];   
                        }
                    } // for n

                    rho[[s,m]]=rsm + cf*(sum_rho-rsm);
                    gama[[s,m]]=gsm + cf*(sum_gama-gsm);
                    omega[[s,m]]=omsm + cf*(sum_omega-omsm);

                    sum_omega=0.0;
                    sum_rho=0.0;
                    sum_gama=0.0; 
                } // for m
            } // for s

        } // free up s_xxx, d1_xxx, and d2_xxx arrays 


        /* CHECK FOR DIVERGENCE */

        if  (omega[[1,0]]).abs() >100.0 || (rho[[1,0]]).abs() > 100.0 || (gama[[1,0]]).abs() >300.0 {
            *a_check= 200; 
            break; 
        }


        /* TREAT SPHERICAL CASE */
        
        if approx::abs_diff_eq!(r_ratio, 1.0) {
            for s in 0 ..= SDIV-1 { // for(s=1;s<=SDIV;s++)
                for m in 0 ..= MDIV-1 { // (m=1;m<=MDIV;m++) {
                    rho[[s,m]]=rho[[s,0]];
                    gama[[s,m]]=gama[[s,0]];
                    omega[[s,m]]=0.0;          
                }
            }
        

        /* TREAT INFINITY WHEN SMAX=1.0 */

        if approx::abs_diff_eq!(SMAX,1.0) {
            for m in 0 ..= MDIV-1 { // (m=1;m<=MDIV;m++) {
                rho[[SDIV-1,m]]=0.0;
                gama[[SDIV-1,m]]=0.0;
                omega[[SDIV-1,m]]=0.0;
                }
            }
        } 

        {
            /* COMPUTE FIRST ORDER DERIVATIVES OF GAMA */ 

            let  da_dm = &mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);
            let  dgds = &mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);
            let  dgdm = &mut Array2::<f64>::zeros((SDIV,MDIV));  // dmatrix(1,SDIV,1,MDIV);



            for s in 0 ..= SDIV-1 { // for(s=1;s<=SDIV;s++)
                for m in 0 ..= MDIV-1 { // (m=1;m<=MDIV;m++) {
                    dgds[[s,m]]=deriv_s(gama,s,m);
                    dgdm[[s,m]]=deriv_m(gama,s,m);
                }
            }


            /* ALPHA */

            if approx::abs_diff_eq!(r_ratio, 1.0, epsilon=f64::EPSILON) {
                for s in 0 ..= SDIV-1 { // for(s=1;s<=SDIV;s++)
                    for m in 0 ..= MDIV-1 { //for(m=1;m<=MDIV;m++)
                        da_dm[[s,m]]=0.0; 
                    }
                }
            } else { 
                for s in 1 ..= SDIV-1 { // for(s=2;s<=SDIV;s++) {
                    for m in 0 ..= MDIV-1 { // for(m=1;m<=MDIV;m++) {

                        da_dm[[0,m]]=0.0; 
                
                        let sgp=s_gp[s];
                        let s1=sgp*(1.0-sgp);
                        let mum=mu[m]; 
                        let m1=1.0-mum.powi(2);
                
                        let d_gama_s=dgds[[s,m]];
                        let d_gama_m=dgdm[[s,m]];
                        let d_rho_s=deriv_s(rho,s,m);
                        let d_rho_m=deriv_m(rho,s,m);
                        let d_omega_s=deriv_s(omega,s,m);
                        let d_omega_m=deriv_m(omega,s,m);
                        let d_gama_ss=s1*deriv_s(dgds,s,m)+(1.0-2.0*sgp)
                                                        *d_gama_s;
                        let d_gama_mm=m1*deriv_m(dgdm,s,m)-2.0*mum*d_gama_m;  
                        let d_gama_sm=deriv_sm(gama,s,m);

                        let temp1=2.0 * sgp.powi(2) * sgp/(1.0-sgp) * m1 * d_omega_s * d_omega_m
                            * (1.0 + s1 * d_gama_s) - (( sgp.powi(2)*d_omega_s).powi(2) - 
                            (sgp*d_omega_m/(1.0-sgp)).powi(2)*m1)*(-mum+m1*d_gama_m); 

                        let temp2=1.0/(m1 * (1.0+s1*d_gama_s).powi(2) + (-mum+m1*d_gama_m).powi(2));

                        let temp3=s1*d_gama_ss + (s1*d_gama_s).powi(2);

                        let temp4=d_gama_m*(-mum+m1*d_gama_m);
                
                        let temp5=( (s1*(d_rho_s+d_gama_s)).powi(2) - m1 * (d_rho_m+d_gama_m).powi(2))
                            *(-mum+m1*d_gama_m);

                        let temp6=s1*m1*(0.5*(d_rho_s+d_gama_s)* (d_rho_m+d_gama_m) 
                            + d_gama_sm + d_gama_s*d_gama_m)*(1.0 + s1*d_gama_s); 

                        let temp7=s1*mum*d_gama_s*(1.0+s1*d_gama_s);

                        let temp8=m1 * (-2.0 * rho[[s,m]]).exp();

                        da_dm[[s,m]] = -0.5*(d_rho_m+d_gama_m) - temp2*(0.5*(temp3 - 
                            d_gama_mm - temp4)*(-mum+m1*d_gama_m) + 0.25*temp5 
                            - temp6 +temp7 + 0.25*temp8*temp1);	 
                    } // for m
                } // for s
            } // else not r_ratio approx 1.0

            for s in 0 ..= SDIV-1 { // for(s=1;s<=SDIV;s++) {
                alpha[[s,0]]=0.0;
                for m in 0 ..= MDIV-2 { // for(m=1;m<=MDIV-1;m++) {
                    alpha[[s,m+1]]=alpha[[s,m]]+0.5*DM*(da_dm[[s,m+1]]+
                                    da_dm[[s,m]]);
                    }
            }
        }


    //  free_dmatrix(da_dm,1,SDIV,1,MDIV);
    //  free_dmatrix(dgds,1,SDIV,1,MDIV);
    //  free_dmatrix(dgdm,1,SDIV,1,MDIV);


        for s in 0 ..= SDIV-1 { // for(s=1;s<=SDIV;s++)
            for m in 0 ..= MDIV-1 { // (m=1;m<=MDIV;m++) {
                alpha[[s,m]] += -alpha[[s,MDIV-1]]+0.5*(gama[[s,MDIV-1]]-rho[[s,MDIV-1]]);

            if alpha[[s,m]]>=300.0 {
                *a_check=200; 
                break;
            }
            omega[[s,m]] /= r_e;
            } 
        }

        if approx::abs_diff_eq!(SMAX,1.0) {
            for m in 0 ..= MDIV-1 { // for(m=1;m<=MDIV;m++)      
                alpha[[SDIV-1,m]] = 0.0;
            }
        }

        if *a_check == 200 {
            break;
        }

        dif=(r_e_old-r_e).abs()/r_e;
        n_of_it += 1;

    }   /* end while */



   /* COMPUTE OMEGA */  

   *big_omega =  match eos_type {
        EosType::Table => big_omega_h * CC / (r_e * KAPPA.sqrt()),
        EosType::Polytropic => big_omega_h / r_e,
    };


    /* UPDATE r_e_new */

    *r_e_new = r_e;


//   free_f3tensor(f_rho, 1,SDIV,1,LMAX+1,1,SDIV);
//   free_f3tensor(f_gama,1,SDIV,1,LMAX+1,1,SDIV);
//   free_dmatrix(p_2n,   1,MDIV,1,LMAX+1);   
//   free_dmatrix(p1_2n_1,1,MDIV,1,LMAX+1);  

} // spin()



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