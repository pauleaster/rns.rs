
use std::{fs::File, io::{BufRead, BufReader}};

use ndarray::{Array, Array1, Axis, array};
use assert_approx_eq::assert_approx_eq;

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
    let (rho, p, h, n0, n_tab) = read_eos_file("./eos/eosA");
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

fn read_eos_file(filename: &str, ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {

    // Ordering of the columns vectors:  rho, p, h, n0
    let mut rho : Vec<f64> = vec![];
    let mut p : Vec<f64> = vec![];
    let mut h : Vec<f64> = vec![];
    let mut n0 : Vec<f64> = vec![];

    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);
    let n_tab = reader.lines()
                            .next()
                            .expect("Unable to read the first line in the EoS file.")
                            .expect("Unable to convert the first line of the EoS file to a string.")
                            .parse::<usize>()
                            .expect("Unable to convert the first line of the EoS file to a number.");
    
    println!("{} lines in EoS file {}", n_tab, filename);

    let mut i: usize = 0;
    let file = File::open(filename).unwrap();
    for res in  csv::ReaderBuilder::new()
                                    .flexible(true)
                                    .from_reader(file)
                                    .records() {
        
        let str_rec = res.expect("Unable to read the EoS file.");
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
    (rho, p, h, n0, n_tab)
}


#[test]
fn test_load_eos () {

    
    let (log_e_tab, log_p_tab, log_h_tab, log_n0_tab, n_tab) = load_eos("./eos/eosA"); 
    // These values measured from the RNS C code
    assert_approx_eq!(log_e_tab[0],-1.340339410862500280e+01);
    assert_approx_eq!(log_p_tab[0],-2.184703547104846777e+01);
    assert_approx_eq!(log_h_tab[0],-2.095363428426100327e+01);
    assert_approx_eq!(log_n0_tab[0], 2.537649832119548066e+01);
    assert_approx_eq!(log_e_tab[n_tab-1], 1.786437651700104601e+00);
    assert_approx_eq!(log_p_tab[n_tab-1], 1.839386676124790565e+00);
    assert_approx_eq!(log_h_tab[n_tab-1], 3.678710661420371286e-01);
    assert_approx_eq!(log_n0_tab[n_tab-1],  3.988153328870259173e+01);
    
}
fn load_eos(filename: &str, ) -> (Array1<f64>, Array1<f64>, Array1<f64>, Array1<f64>, usize) {
    
    let (rho, p, h, n0, n_tab) = read_eos_file(filename); 

    let mut log_e_tab = (Array1::from_vec(rho) * (CC * CC * KSCALE)).mapv(f64::log10);
    let mut log_p_tab = (Array1::from_vec(p) * KSCALE).mapv(f64::log10);
    let mut log_h_tab = (Array1::from_vec(h) / (CC*CC)).mapv(f64::log10);
    let mut log_n0_tab = Array1::from_vec(n0).mapv(f64::log10);

    (log_e_tab, log_p_tab, log_h_tab, log_n0_tab, n_tab)
}

/*******************************************************************/
fn e_of_rho0(rho0: f64, gamma_p: f64) -> f64 {
    rho0.powf(gamma_p)/(gamma_p-1.0)+rho0
}