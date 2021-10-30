use std::cmp::{min,max};
use assert_approx_eq::assert_approx_eq;

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
        let z = hunt(&y1, y_ref_val);
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
        let z = hunt(&y1, y_ref_val);
        println!("z={}",z);
        println!("{} == {}? = {}",result_val, z, result_val ==z );
        assert!((z as f64 - result_val as f64).abs() <= 1.0);
    }

}

/***************************************************************************/
/* Routine that locates nearest grid point for a given value.              */
/* Adapted from Numerical Recipes.                                         */
/***************************************************************************/
fn hunt(xx: &[f64], x: f64) -> usize {

	let n = xx.len();
    let mut jlo = n >> 1;
    
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
    
}

/*************************************************************************/
/* Driver for the interpolation routine. First we find the tab. point    */
/* nearest to xb, then we interpolate using four points around xb.       */  
/*************************************************************************/
fn interp(  xp: &[f64], 
            yp: &[f64], 
            xb: f64) -> f64 { 

    let m=4;      /* degree of interpolation */ 

    let nearest = hunt(xp,xb);

    let np = xp.len();
    let k=max(0,
                    min(
                        max(nearest as i32 - ((m-1)>>1) as i32,
                            1),
                                np as i32 - m as i32)) as usize;

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