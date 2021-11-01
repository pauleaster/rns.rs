
use std::{fs::File, io::{BufRead, BufReader}};

use ndarray::{Array, Array1, Axis, array};
use assert_approx_eq::assert_approx_eq;
use std::error::Error;

use crate::equil_util::{EosType,interp};

use csv;

use crate::consts::*;

#[test]
fn test_make_grid() {
    println!("Using default constants: SDIV={}, MDIV={}",SDIV,MDIV);
    let (s,m) = make_grid(None,None);
    println!("s[0]={:?}, s[{}]={:?}",s[0],SDIV,s[SDIV-1]);
    println!("m[0]={:?}, s[{}]={:?}",m[0],MDIV,m[MDIV-1]);
    assert_approx_eq!(s[0]-0.0,f64::EPSILON);
    assert_approx_eq!(s[SDIV-1],SMAX,f64::EPSILON);
    assert_approx_eq!(m[0],0.0,f64::EPSILON);
    assert_approx_eq!(m[MDIV-1],1.0,f64::EPSILON);
    let sdiv = 10_usize;
    let mdiv = 20_usize;
    println!("Using default constants: sdiv={}, mdiv={}",sdiv,mdiv);
    let (s,m) = make_grid(Some(sdiv),Some(mdiv));
    println!("s[0]={:?}, s[{}]={:?}",s[0],sdiv,s[sdiv-1]);
    println!("m[0]={:?}, s[{}]={:?}",m[0],mdiv,m[mdiv-1]);
    assert_approx_eq!(s[0]-0.0,f64::EPSILON);
    assert_approx_eq!(s[sdiv-1],SMAX,f64::EPSILON);
    assert_approx_eq!(m[0],0.0,f64::EPSILON);
    assert_approx_eq!(m[mdiv-1],1.0,f64::EPSILON);

}

fn make_grid(s_size: Option<usize>, mu_size: Option<usize>) -> (Array1<f64>,Array1<f64>){

    let s_dim = match s_size {
        Some(s) => s as f64,
        None => SDIV as f64,
    };
    let mu_dim = match mu_size {
        Some(m) => m as f64,
        None => MDIV as f64,
    };
    let s_gp = Array1::range(0., s_dim, 1.0) * (SMAX / (s_dim - 1.0));
    let mu = Array1::range(0., mu_dim, 1.0) / (mu_dim - 1.0);
    (s_gp, mu)

}

#[test]
fn test_read_eos_file () {

    // Ordering of the columns vectors:  rho, p, h, n0
    let (rho, p, h, n0, n_tab) = read_eos_file("./eos/eosA").unwrap();
    assert_eq!(n_tab, 102);
    // First row:
    // 3.95008e+01 1.27820e+14 1.000000000000000e+00 2.379569102499467e+25 
    assert_approx_eq!(rho[0], 3.95008e+01);
    assert_approx_eq!(p[0], 1.27820e+14);
    assert_approx_eq!(h[0], 1.000000000000000e+00);
    assert_approx_eq!(n0[0], 2.379569102499467e+25 );
    // Last row:
    // 6.11558e+16 6.20899e+37 2.096550609129587e+21 7.612604874394090e+39 
    assert_approx_eq!(rho[n_tab-1], 6.11558e+16);
    assert_approx_eq!(p[n_tab-1], 6.20899e+37);
    assert_approx_eq!(h[n_tab-1], 2.096550609129587e+21);
    assert_approx_eq!(n0[n_tab-1], 7.612604874394090e+39 );
}

fn read_eos_file(filename: &str, ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize), Box<dyn Error>> {

    // Ordering of the columns vectors:  rho, p, h, n0
    let mut rho : Vec<f64> = vec![];
    let mut p : Vec<f64> = vec![];
    let mut h : Vec<f64> = vec![];
    let mut n0 : Vec<f64> = vec![];




    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let n_tab = reader.lines()
                            .next()
                            .expect("Unable to read the first line in the EoS file.")?
                            .parse::<usize>()?;

    println!("{} lines in EoS file {}", n_tab, filename);

    let mut i: usize = 0;
    let file = File::open(filename).unwrap();
    for res in  csv::ReaderBuilder::new()
                                    .flexible(true)
                                    .from_reader(file)
                                    .records() {
        
        let str_rec = res?;
        for str_row in str_rec.iter() {
            i += 1;
            for (idx, str_val) in str_row.split(' ').enumerate() {
                if !str_val.is_empty() {
                    let val_parse = str_val.parse::<f64>();
                    if let Ok(val) = val_parse {
                        // Ordering of the columns vectors:  rho, p, h, n0
                        match idx {
                            0 => rho.push(val),
                            1 => p.push(val),
                            2 => h.push(val),
                            3 => n0.push(val),
                            _ => (),
                        }
                    } else {
                        panic!("Failed to read EoS file in line {} and column {}",i + 1 , idx + 1);
                    }
                }
            }
        }
    }
// Ordering of the columns vectors:  rho, p, h, n0
    Ok((rho, p, h, n0, n_tab))
}


#[test]
fn test_load_eos () {

    
    let (log_e_tab, log_p_tab, log_h_tab, log_n0_tab, n_tab) = load_eos("./eos/eosA").unwrap(); 
    // These values measured from the RNS C code
    assert_approx_eq!(log_e_tab[0],-1.340_339_410_862_500_3e1);
    assert_approx_eq!(log_p_tab[0],-2.184_703_547_104_846_8e1);
    assert_approx_eq!(log_h_tab[0],-2.095_363_428_426_100_3e1);
    assert_approx_eq!(log_n0_tab[0], 2.537_649_832_119_548e1);
    assert_approx_eq!(log_e_tab[n_tab-1], 1.786_437_651_700_104_6e0);
    assert_approx_eq!(log_p_tab[n_tab-1], 1.839_386_676_124_790_6e0);
    assert_approx_eq!(log_h_tab[n_tab-1], 3.678_710_661_420_371_3e-1);
    assert_approx_eq!(log_n0_tab[n_tab-1],  3.988_153_328_870_259e1);
    
}
fn load_eos(filename: &str, ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, usize), Box<dyn Error>> {
    
    let (rho, p, h, n0, n_tab) = read_eos_file(filename)?; 

    let log_e_tab = (Array1::from_vec(rho) * (CC * CC * KSCALE)).mapv(f64::log10);
    let log_p_tab = (Array1::from_vec(p) * KSCALE).mapv(f64::log10);
    let log_h_tab = (Array1::from_vec(h) / (CC*CC)).mapv(f64::log10);
    let log_n0_tab = Array1::from_vec(n0).mapv(f64::log10);

    Ok((log_e_tab, log_p_tab, log_h_tab, log_n0_tab, n_tab))
}

#[test]
fn test_e_of_rho0() { // Should use scaled values here!!
    let rho0 = 5e14;
    let gamma_p = 1.03;
    let result = 4.650_635_748_754_89E16;
    assert_approx_eq!(e_of_rho0(rho0, gamma_p),result,1000.0);
}


fn e_of_rho0(rho0: f64, gamma_p: f64) -> f64 {
    rho0.powf(gamma_p)/(gamma_p-1.0)+rho0
}

#[test]
fn test_e_at_p() {

    {
        let (log_e_tab, 
        log_p_tab, _, _, _) = load_eos("./eos/eosA").unwrap(); 
        // println!("{:?}",log_e_tab);
        // println!("{:?}",log_p_tab);
        let eostype = EosType::Table;
        let e = e_at_p(10.0_f64.powf(0.8), Some(log_e_tab), Some(log_p_tab), eostype, None, Some(0)).unwrap();
        assert_approx_eq!(e, 8.635_331_229_226_669);
    }
    {
        let op_gamma_p = Some(2.025_538_636_879_4);
        let eostype = EosType::Polytropic;
        let e = e_at_p(10.0_f64.powf(0.8), None, None, eostype, op_gamma_p, None).unwrap();
        assert_approx_eq!(e, 8.635_331_229_226_669, 0.000003);
    }
}


fn  e_at_p( pp: f64, 
    opt_log_e_tab: Option<Array1<f64>>, 
    opt_log_p_tab: Option<Array1<f64>>,
    eos_type: EosType,
    op_gamma_p: Option<f64>,
    opt_nearest: Option<usize>) -> Result<f64, Box<dyn Error>> {

    match eos_type {
        EosType::Table => {
            if let Some(log_e_tab) = opt_log_e_tab { 
                if let Some(log_p_tab) = opt_log_p_tab {
                    return Ok(10.0_f64.powf(interp(&log_p_tab.to_vec(),&log_e_tab.to_vec(),pp.log10(),opt_nearest)));
                };
            };
            Err("Using a tabulated EoS but failed to supply both log_e and log_p.".into())
        },
        EosType::Polytropic => {
            if let Some(gamma_p) = op_gamma_p {
                return Ok(pp/(gamma_p-1.0) + pp.powf(1.0/gamma_p));
            };
            Err("Using polytropic EoS but failed to supply gamma_p.".into())
        },
    }
}

#[test]
fn test_p_at_e() {
    let (log_e_tab, 
    log_p_tab, _, _, _) = load_eos("./eos/eosA").unwrap(); 
    
    let e = 8.635_331_229_226_669;
    let pres = 10.0_f64.powf(0.8);
    let p = p_at_e(e, log_e_tab, log_p_tab, Some(44));
    assert_approx_eq!(p, pres,0.002);
    /* Note, because of the interpolation, the inverse for p(e) is different than e(p) by
        delta_p \sim 0.0012 */
}

fn p_at_e(ee: f64, 
    log_e_tab: Array1<f64>, 
    log_p_tab: Array1<f64>,
    opt_nearest: Option<usize>) -> f64 {

    10.0_f64.powf(interp(&log_e_tab.to_vec(),
                        &log_p_tab.to_vec(),
                        ee.log10(),
                        opt_nearest))
} 

// /*C*/
// /*******************************************************************/
// double p_at_h(double hh, 
//     double log_p_tab[201], 
//     double log_h_tab[201],
//     int    n_tab, 
//     int    *n_nearest_pt)
// {
// return pow(10.0,interp(log_h_tab,log_p_tab,n_tab,log10(hh), n_nearest_pt));
// }

// /*C*/
// /*******************************************************************/
// double h_at_p(double pp, 
//     double log_h_tab[201], 
//     double log_p_tab[201],
//     int    n_tab, 
//     int    *n_nearest_pt)
// {
// return pow(10.0,interp(log_p_tab,log_h_tab,n_tab,log10(pp), n_nearest_pt));
// }

// /*C*/
// /*******************************************************************/
// double n0_at_e(double ee, 
//      double log_n0_tab[201], 
//      double log_e_tab[201],
//      int    n_tab, 
//      int    *n_nearest_pt)
// {
// return pow(10.0,interp(log_e_tab,log_n0_tab,n_tab,log10(ee), n_nearest_pt));
// }
