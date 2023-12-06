#![allow(unused)]

use std::cmp::Ordering;

use crate::def::*;

pub mod rnd {
    static mut S: usize = 88172645463325252;

    #[inline]
    pub fn next() -> usize {
        unsafe {
            S = S ^ S << 7;
            S = S ^ S >> 9;
            S
        }
    }

    #[inline]
    pub fn nextf() -> f64 {
        (next() & 4294967295) as f64 / 4294967296.
    }

    #[inline]
    pub fn gen_range(low: usize, high: usize) -> usize {
        assert!(low < high);
        (next() % (high - low)) + low
    }

    pub fn shuffle<I>(vec: &mut Vec<I>) {
        for i in 0..vec.len() {
            let j = gen_range(0, vec.len());
            vec.swap(i, j);
        }
    }
}

pub mod time {
    static mut START: f64 = -1.;
    pub fn start_clock() {
        let _ = elapsed_seconds();
    }

    #[inline]
    pub fn elapsed_seconds() -> f64 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        unsafe {
            if START < 0. {
                START = t;
            }
            t - START
        }
    }
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy)]
pub struct FloatIndex(pub f64);

impl Ord for FloatIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Eq for FloatIndex {}

pub fn calc_score(input: &Input, answer: &Vec<(usize, usize)>) -> i64 {
    let mut last_visited = vec![vec![!0; input.n]; input.n];
    let mut v = vec![];
    let mut average = vec![vec![0.; input.n]; input.n];
    let mut edge_count = vec![vec![(0, 0); input.n]; input.n];
    for i in 0..answer.len() {
        last_visited[answer[i].0][answer[i].1] = i;
    }
    let mut s = 0;
    let mut sum_d = 0;
    for i in 0..input.n {
        for j in 0..input.n {
            s += (answer.len() - last_visited[i][j]) as i64 * input.d[i][j];
            sum_d += input.d[i][j];
        }
    }
    let mut last_visited2 = last_visited.clone();
    let mut sum = vec![vec![0; input.n]; input.n];
    for t in answer.len()..2 * answer.len() {
        let (i, j) = answer[t - answer.len()];
        let dt = (t - last_visited2[i][j]) as i64;
        let a = dt * input.d[i][j];
        sum[i][j] += dt * (dt - 1) / 2 * input.d[i][j];
        s -= a;
        last_visited2[i][j] = t;
        v.push(s);
        s += sum_d;
    }
    for i in 0..input.n {
        for j in 0..input.n {
            average[i][j] = sum[i][j] as f64 / answer.len() as f64;
        }
    }
    let score = (2 * v.iter().sum::<i64>() + answer.len() as i64) / (2 * answer.len()) as i64;
    score
}
