#![allow(dead_code)]

use serde::Serialize;

pub fn msb(x: u64) -> u8 {
    let bits = 64u8;
    if x != 0 {
        bits - 1 - (x.leading_zeros() as u8)
    } else {
        0u8
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Stats {
    pub count: usize,
    pub max: u128,
    pub min: u128,
    pub mean: f64,
    pub variance: f64,
}

pub fn compute_stats_u128<I>(iter: I) -> Stats
where
    I: Iterator<Item = u128>,
{
    let mut sum: u128 = 0;
    let mut sum_square: u128 = 0;
    let mut count: usize = 0;
    let mut min = u128::max_value();
    let mut max = 0u128;

    for v in iter {
        sum += v as u128;
        sum_square += (v as u128) * (v as u128);
        count += 1;
        min = min.min(v as u128);
        max = max.max(v as u128);
    }

    let mean = if count > 0 {
        (sum as f64) / (count as f64)
    } else {
        0.0
    };

    let variance = if count > 0 {
        ((sum_square as f64) / (count as f64)) - (mean * mean)
    } else {
        0.0
    };

    Stats {
        count,
        max,
        min,
        mean,
        variance,
    }
}
pub fn compute_stats<I>(iter: I) -> Stats
where
    I: Iterator<Item = usize>,
{
    compute_stats_u128(iter.map(|val| val as u128))
}

pub fn compute_modes<I>(iter: I, max: usize) -> Vec<usize>
where
    I: Iterator<Item = usize>,
{
    let mut modes = vec![0usize; max + 1];
    for v in iter {
        modes[v] += 1;
    }

    modes
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct ModeStats(pub usize, pub usize, pub f64, pub f64); // (min,max,mean,var)

#[derive(Debug, Clone, Copy, Serialize)]
struct ModeStatsAux(usize, usize, usize, usize); // (min,max,sum,sum_square)

pub fn compute_modes_stat<'a, I>(iter: I, max: usize) -> Vec<ModeStats>
// (Vec<f64>,Vec<f64>)
where
    I: Iterator<Item = &'a Vec<usize>>,
{
    // let mut modes_sum = vec![0; max + 1];
    // let mut modes_square = vec![0; max + 1];
    // let mut modes_min = vec![usize::max_value(); max + 1];
    // let mut modes_max = vec![0; max + 1];

    let mut modes_stats = vec![ModeStatsAux(usize::max_value(), 0, 0, 0); max + 1];

    let mut count = 0;

    for m in iter {
        for i in 0..(m.len().min(max + 1)) {
            let ModeStatsAux(min, max, sum, sum_square) = modes_stats[i];

            modes_stats[i] = ModeStatsAux(
                min.min(m[i]),
                max.max(m[i]),
                sum + m[i],
                sum_square + (m[i] * m[i]),
            );
        }
        count += 1;
    }

    if count == 0 {
        return Vec::<ModeStats>::new();
    }

    modes_stats
        .into_iter()
        .map(|ModeStatsAux(min, max, sum, sum_square)| {
            let mean = (sum as f64) / f64::from(count);
            ModeStats(
                min,
                max,
                mean,
                ((sum_square as f64) / f64::from(count)) - mean * mean,
            )
        })
        .collect()
}

pub fn compute_overflow_stat<'a, I>(iter: I, overflow_limit: usize) -> usize
where
    I: Iterator<Item = &'a usize>,
{
    (0..)
        .zip(iter)
        .map(|(i, x)| {
            if i > overflow_limit {
                (i - overflow_limit) * x
            } else {
                0
            }
        })
        .sum()
}

pub trait Mean<A = Self>: Sized {
    fn mean<I: Iterator<Item = A>>(iter: I) -> f64;
}

macro_rules! unsigned_integer_mean {
    (@impls  $($a:ty)*) => ($(
        impl Mean for $a {
            fn mean<I: Iterator<Item=$a>>(iter: I) -> f64 {
                let mut sum: u128 = 0;
                let mut count: usize = 0;

                for v in iter {
                    sum += v as u128;
                    count += 1;
                }
                if count > 0 {
                    (sum as f64) / (count as f64)
                } else {
                    0.0
                }
            }
        }

        impl<'a> Mean<&'a $a> for $a {
            fn mean<I: Iterator<Item=&'a $a>>(iter: I) -> f64 {
                let mut sum: u128 = 0;
                let mut count: usize = 0;

                for v in iter {
                    sum += *v as u128;
                    count += 1;
                }
                if count > 0 {
                    (sum as f64) / (count as f64)
                } else {
                    0.0
                }
            }
        }
    )*);
    ($($a:ty)*) => (
        unsigned_integer_mean!(@impls $($a)+);
    );
}

macro_rules! signed_integer_mean {
    (@impls $($a:ty)*) => ($(
        impl Mean for $a {
            fn mean<I: Iterator<Item=$a>>(iter: I) -> f64 {
                let mut sum: i128 = 0;
                let mut count: usize = 0;

                for v in iter {
                    sum += v as i128;
                    count += 1;
                }
                if count > 0 {
                    (sum as f64) / (count as f64)
                } else {
                    0.0
                }
            }
        }

        impl<'a> Mean<&'a $a> for $a {
            fn mean<I: Iterator<Item=&'a $a>>(iter: I) -> f64 {
                let mut sum: i128 = 0;
                let mut count: usize = 0;

                for v in iter {
                    sum += *v as i128;
                    count += 1;
                }
                if count > 0 {
                    (sum as f64) / (count as f64)
                } else {
                    0.0
                }
            }
        }
    )*);
    ($($a:ty)*) => (
        signed_integer_mean!(@impls $($a)+);
    );
}

macro_rules! float_mean {
    (@impls $($a:ty)*) => ($(
        impl Mean for $a {
            fn mean<I: Iterator<Item=$a>>(iter: I) -> f64 {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;

                for v in iter {
                    sum += v as f64;
                    count += 1;
                }
                if count > 0 {
                    sum / (count as f64)
                } else {
                    0.0
                }
            }
        }

        impl<'a> Mean<&'a $a> for $a {
            fn mean<I: Iterator<Item=&'a $a>>(iter: I) -> f64 {
                let mut sum: f64 = 0.0;
                let mut count: usize = 0;

                for v in iter {
                    sum += *v as f64;
                    count += 1;
                }
                if count > 0 {
                    sum / (count as f64)
                } else {
                    0.0
                }
            }
        }
    )*);
    ($($a:ty)*) => (
        float_mean!(@impls $($a)+);
    );
}

unsigned_integer_mean! { u8 u16 u32 u64 u128 usize }
signed_integer_mean! { i8 i16 i32 i64 i128 isize }
float_mean! { f32 f64 }
