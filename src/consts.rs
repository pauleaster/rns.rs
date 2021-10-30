
pub const SDIV : usize = 129;
pub const MDIV: usize = 65;
pub const SMAX: f64 = 0.9999;
pub const CC:f64 =  2.9979e10;        /* speed of light in vacuum */
pub const GG: f64 =  6.6732e-8;       /* gravitational constant */ 
pub const KAPPA: f64 =  1.0e-15*CC*CC/GG;  /* scaling factor */
pub const KSCALE: f64 = KAPPA*GG/(CC*CC*CC*CC);   /* another scaling factor */
pub const MSUN: f64 = 1.987e33;       /* Mass of Sun */