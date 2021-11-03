use std::f64::consts::PI;


pub const RDIV: usize = 900;
pub const SDIV : usize = 129;
pub const SDIV_MIN_1 : usize = SDIV - 1; // for limitations in match
pub const SDIV_MIN_2 : usize = SDIV - 2; // for limitations in match
pub const SDIV_MIN_3 : usize = SDIV - 3; // for limitations in match
pub const MDIV: usize = 65;
pub const MDIV_MIN_1 : usize = MDIV - 1; // for limitations in match
pub const MDIV_MIN_2 : usize = MDIV - 2; // for limitations in match
pub const MDIV_MIN_3 : usize = MDIV - 3; // for limitations in match
pub const SMAX: f64 = 0.9999;
pub const DM: f64 = 1./((MDIV  - 1) as f64);
pub const DMSQ: f64 = DM * DM;
pub const DS: f64 = (SMAX as f64)/((SDIV  - 1) as f64);
pub const DSSQ: f64 = DS * DS;
pub const CC:f64 =  2.9979e10;        /* speed of light in vacuum */
pub const GG: f64 =  6.6732e-8;       /* gravitational constant */ 
pub const KAPPA: f64 =  1.0e-15*CC*CC/GG;  /* scaling factor */
pub const KSCALE: f64 = KAPPA*GG/(CC*CC*CC*CC);   /* another scaling factor */
pub const MSUN: f64 = 1.987e33;       /* Mass of Sun */
pub const MAXIT: usize = 100;
pub const RMIN: f64 = 1e-15;
