
use std::{fs::File, process::exit};

use ndarray::{Array, Array1, array};
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
fn test_load_eos () {
    let le : Array1<f64> = array![];
    let lp : Array1<f64> = array![];
    let lh : Array1<f64> = array![];
    let ln0 : Array1<f64> = array![];
    let n : Array1<usize> = array![];
    load_eos("./eos/eosA",le,lp,lh,ln0,n);


    
}

fn load_eos(filename: &str, 
    log_e_tab: Array1<f64>,
    log_p_tab: Array1<f64>,
    log_h_tab: Array1<f64>,
    log_n0_tab: Array1<f64>, 
    n_tab: Array1<usize>) {

        let mut file = File::open(filename).unwrap();
        let mut reader = csv::ReaderBuilder::new()
                                        .delimiter(b' ')
                                        .from_reader(file);
        let mut i:usize = 0;
        for result in reader.records() {
            i += 1;
            // The iterator yields Result<StringRecord, Error>, so we check the
            // error here.
            let record_result = result;
            
            let record = match record_result {
                Ok(r) => r,
                Err(e) => {
                    println!("Error in record on line {}",i);
                    println!("Error: {:?}",e);
                    continue;
                }
            };
            println!("{}: {:?}", i,record);
        }

    }