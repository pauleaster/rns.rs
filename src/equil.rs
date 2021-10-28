
use ndarray::{Array1,Array};
use assert_approx_eq::assert_approx_eq;


use crate::consts::*;


#[test]
fn test_make_grid() {
    let (s,m) = make_grid();
    println!("s[0]={:?}, s[{}]={:?}",s[0],SDIV,s[SDIV-1]);
    println!("m[0]={:?}, s[{}]={:?}",m[0],MDIV,m[MDIV-1]);
    assert_approx_eq!(s[0]-0.0,f64::EPSILON);
    assert_approx_eq!(s[SDIV-1],SMAX,f64::EPSILON);
    assert_approx_eq!(m[0],0.0,f64::EPSILON);
    assert_approx_eq!(m[MDIV-1],1.0,f64::EPSILON);

}

fn make_grid() -> (Array1::<f64>,Array1::<f64>){

    let s_gp = Array1::range(0., SDIV as f64, 1.0) * (SMAX / (SDIV as f64 - 1.0));
    let mu = Array1::range(0., MDIV as f64, 1.0) / (MDIV as f64 - 1.0);
    (s_gp, mu)

}