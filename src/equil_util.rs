use std::cmp::max;

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
fn hunt(xx: &Vec<f64>, x: f64) -> usize {

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

// /*C*/
// /*************************************************************************/
// /* Driver for the interpolation routine. First we find the tab. point    */
// /* nearest to xb, then we interpolate using four points around xb.       */  
// /*************************************************************************/
// double interp(double xp[], 
//               double yp[], 
//               int    np ,
//               double xb, 
//               int    *n_nearest_pt)
// { 
//  int k,        /* index of 1st point */
//      m=4;      /* degree of interpolation */ 
 
//  double y;     /* intermediate value */

//  hunt(xp,np,xb,n_nearest_pt);

//  k=IMIN(IMAX((*n_nearest_pt)-(m-1)/2,1),np+1-m);

//  if( xb==xp[k] ||  xb==xp[k+1] || xb==xp[k+2] || xb==xp[k+3]) 
//     xb += DBL_EPSILON;

//  y= (xb-xp[k+1])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k]/
//         ((xp[k]-xp[k+1])*(xp[k]-xp[k+2])*(xp[k]-xp[k+3]))
 
//     +(xb-xp[k])*(xb-xp[k+2])*(xb-xp[k+3])*yp[k+1]/
//        ((xp[k+1]-xp[k])*(xp[k+1]-xp[k+2])*(xp[k+1]-xp[k+3]))
 
//     +(xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+3])*yp[k+2]/
//        ((xp[k+2]-xp[k])*(xp[k+2]-xp[k+1])*(xp[k+2]-xp[k+3]))
 
//     +(xb-xp[k])*(xb-xp[k+1])*(xb-xp[k+2])*yp[k+3]/
//        ((xp[k+3]-xp[k])*(xp[k+3]-xp[k+1])*(xp[k+3]-xp[k+2]));

//  return (y);
// }