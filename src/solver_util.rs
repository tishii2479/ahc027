use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::util::*;

pub type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

pub fn solve_tsp(
    v: &Vec<(usize, usize)>,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    iter_cnt: usize,
) -> (Vec<usize>, i64) {
    let n = v.len();
    let mut order: Vec<usize> = (0..n).map(|x| x).collect();
    let mut score = 0;
    for i in 0..n {
        let j = (i + 1) % n;
        score += dist[v[i].0][v[i].1][v[j].0][v[j].1];
    }

    for _t in 0..=iter_cnt {
        let (a, b) = (rnd::gen_range(0, n), rnd::gen_range(0, n));
        if a == b {
            continue;
        }
        let (from_a, from_b) = (order[a], order[b]);
        let (to_a, to_b) = (order[(a + 1) % n], order[(b + 1) % n]);
        let score_delta = dist[v[from_a].0][v[from_a].1][v[from_b].0][v[from_b].1]
            + dist[v[to_a].0][v[to_a].1][v[to_b].0][v[to_b].1]
            - dist[v[from_a].0][v[from_a].1][v[to_a].0][v[to_a].1]
            - dist[v[from_b].0][v[from_b].1][v[to_b].0][v[to_b].1];

        if score_delta <= 0 {
            // before:  from_i -> to_i -> b -> a -> from_j -> to_j -> ... -> i
            // after:   from_i -> j -> a -> b -> to[i] -> to_j -> ... -> i
            let (mut x, mut y) = if a < b { (a + 1, b) } else { (b + 1, a) };
            while x < y {
                order.swap(x, y);
                x += 1;
                y -= 1;
            }
            score += score_delta;
        }

        // if _t % (iter_cnt / 10) == 0 {
        //     eprintln!("{} {}", _t, score);
        // }
    }

    (order, score)
}

pub fn get_prev_and_next(x: FloatIndex, btree: &BTreeSet<FloatIndex>) -> (f64, f64) {
    let first = btree.iter().next().unwrap().0 + 2.;
    let last = btree.iter().next_back().unwrap().0 - 2.;
    let prev = btree
        .range((std::ops::Bound::Unbounded, std::ops::Bound::Excluded(x)))
        .next_back()
        .unwrap_or(&FloatIndex(last))
        .0;
    let next = btree
        .range((std::ops::Bound::Excluded(x), std::ops::Bound::Unbounded))
        .next()
        .unwrap_or(&FloatIndex(first))
        .0;
    (prev, next)
}

pub fn calc_prev_delta(x: FloatIndex, btree: &BTreeSet<FloatIndex>) -> f64 {
    let (prev, _) = get_prev_and_next(x, btree);
    assert!(prev <= x.0 && x.0 < 2.);
    ((x.0 - prev) as f64).powf(2.)
}

/// xをbtreeに追加した時のスコアの差分を計算する
pub fn calc_delta(x: FloatIndex, btree: &BTreeSet<FloatIndex>) -> f64 {
    let (prev, next) = get_prev_and_next(x, btree);
    assert!(prev <= x.0 && x.0 <= next);
    (((x.0 - prev) as f64).powf(2.) + ((next - x.0) as f64).powf(2.))
        - ((next - prev) as f64).powf(2.)
}

pub fn cycles_to_path(cycles: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut path = vec![];
    for cycle in cycles {
        path.extend(cycle.clone());
    }
    let start = path
        .iter()
        .position(|&v| v == (0, 0))
        .expect("No (0, 0) in cycle");

    (start..start + path.len())
        .map(|i| path[i % path.len()])
        .collect()
}

pub fn create_adj(input: &Input) -> Adj {
    let mut adj = vec![vec![vec![]; input.n]; input.n];
    for j in 0..input.n {
        for i in 0..input.n {
            // 上
            if i > 0 && !input.h[i - 1][j] {
                adj[i][j].push((Dir::Up, (i - 1, j)));
            }
            // 右
            if j < input.n - 1 && !input.v[i][j] {
                adj[i][j].push((Dir::Right, (i, j + 1)));
            }
            // 下
            if i < input.n - 1 && !input.h[i][j] {
                adj[i][j].push((Dir::Down, (i + 1, j)));
            }
            // 左
            if j > 0 && !input.v[i][j - 1] {
                adj[i][j].push((Dir::Left, (i, j - 1)));
            }
        }
    }
    adj
}

pub fn calc_r(input: &Input) -> Vec<Vec<f64>> {
    let mut a_sum = 0.;
    let mut r = vec![vec![0.; input.n]; input.n];
    for i in 0..input.n {
        for j in 0..input.n {
            r[i][j] = (input.d[i][j] as f64).powf(1. / 3.);
            a_sum += r[i][j];
        }
    }
    for i in 0..input.n {
        for j in 0..input.n {
            r[i][j] /= a_sum;
        }
    }
    r
}

pub fn calc_dist(s: (usize, usize), input: &Input, adj: &Adj) -> Vec<Vec<i64>> {
    let (s_i, s_j) = s;
    let mut dist = vec![vec![1 << 30; input.n]; input.n];
    dist[s_i][s_j] = 0;
    let mut q = VecDeque::new();
    q.push_back((s_i, s_j));
    while let Some((v_i, v_j)) = q.pop_front() {
        for (_, (nxt_i, nxt_j)) in adj[v_i][v_j].iter() {
            if dist[*nxt_i][*nxt_j] <= dist[v_i][v_j] + 1 {
                continue;
            }
            dist[*nxt_i][*nxt_j] = dist[v_i][v_j] + 1;
            q.push_back((*nxt_i, *nxt_j));
        }
    }
    dist
}

pub fn calc_gain(t1: f64, t2: f64, d: i64) -> f64 {
    (t1 - t2).powf(2.) * d as f64
}

#[allow(unused)]
pub fn show(cycles: &Vec<Vec<(usize, usize)>>, input: &Input) {
    eprintln!("-----");
    let mut counts = vec![vec![0; input.n]; input.n];
    for cycle in cycles.iter() {
        for v in cycle.iter() {
            counts[v.0][v.1] += 1;
        }
    }

    for i in 0..counts.len() {
        for j in 0..counts[i].len() {
            eprint!("{:4}", counts[i][j]);
        }
        eprintln!();
    }
}

#[allow(unused)]
pub fn show_path(path: &Vec<(usize, usize)>, n: usize) {
    let mut a = vec![vec![0; n]; n];
    for (i, j) in path {
        a[*i][*j] += 1;
    }
    for i in 0..n {
        for j in 0..n {
            eprint!("{:2}", a[i][j]);
        }
        eprintln!();
    }
}

#[test]
pub fn test_calc_delta() {
    let mut btree = BTreeSet::new();
    let total_cycle_length = 1000000;
    for _ in 0..10 {
        btree.insert(FloatIndex(rnd::gen_range(0, total_cycle_length) as f64));
    }
    pub fn calc_actual_score(btree: &BTreeSet<FloatIndex>) -> f64 {
        let mut score = 0.;
        for x in btree.iter() {
            let (prev, next) = get_prev_and_next(*x, btree);
            score += ((x.0 - prev).powf(2.) + (x.0 - next).powf(2.)) / 2.;
        }
        score
    }
    let mut score = calc_actual_score(&btree);
    for _ in 0..10000 {
        let prev_score = score;
        let x = FloatIndex(rnd::gen_range(0, total_cycle_length) as f64);
        if btree.contains(&x) {
            continue;
        }

        btree.insert(x);
        score += calc_delta(x, &btree);
        assert_eq!(score, calc_actual_score(&btree));

        score -= calc_delta(x, &btree);
        btree.remove(&x);
        assert_eq!(prev_score, score);
    }
}
