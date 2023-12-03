use std::collections::{BTreeMap, VecDeque};

use crate::def::*;

type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

fn create_adj(input: &Input) -> Adj {
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

fn calc_r(input: &Input) -> Vec<Vec<f64>> {
    let mut a_sum = 0.;
    let mut r = vec![vec![0.; input.n]; input.n];
    for i in 0..input.n {
        for j in 0..input.n {
            r[i][j] = r[0][0] * (input.d[i][j] as f64).powf(1. / 3.);
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

fn calc_dist(s: (usize, usize), input: &Input, adj: &Adj) -> Vec<Vec<i64>> {
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

fn get_move_cand(
    s: (usize, usize),
    t: (usize, usize),
    dist: &Vec<Vec<i64>>,
    adj: &Adj,
) -> Vec<Dir> {
    assert_ne!(s, t);
    let mut q = VecDeque::new();
    q.push_back((Dir::Down, t)); // Dir::Downはダミーの値

    let mut cand = vec![];
    while let Some((dir, (v_i, v_j))) = q.pop_front() {
        if dist[v_i][v_j] == 0 {
            cand.push(dir.inv());
            continue;
        }
        for (dir, (nxt_i, nxt_j)) in adj[v_i][v_j].iter() {
            if dist[*nxt_i][*nxt_j] != dist[v_i][v_j] - 1 {
                continue;
            }
            if q.contains(&(*dir, (*nxt_i, *nxt_j))) {
                continue;
            }
            q.push_back((*dir, (*nxt_i, *nxt_j)));
        }
    }

    cand
}

fn get_move_cand(
    s: (usize, usize),
    t: (usize, usize),
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    adj: &Adj,
) -> Vec<Dir> {
    vec![]
}

pub fn solve2(input: &Input) -> String {
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let (mut s_i, mut s_j) = (0, 0);
    let mut counts: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);

            let t = (1e5 * r[i][j]).round() as usize;
            if let Some(count) = counts.get_mut(&t) {
                count.push((i, j));
            } else {
                counts.insert(t, vec![(i, j)]);
            }

            if r[i][j] > r[s_i][s_j] {
                (s_i, s_j) = (i, j);
            }
        }
    }

    let (s_i, s_j) = (s_i, s_j);
    let cycle_l = (1. / r[s_i][s_j]).round() as usize;
    let cycle_cnt = (1e5 as usize) / cycle_l;
    let mut cycles = vec![];

    for _ in 0..cycle_cnt {
        let (mut v_i, mut v_j) = (s_i, s_j);
        let mut path = vec![];
        while cycle_l - path.len() > dist[s_i][s_j][v_i][v_j] as usize {
            let next_v = counts
                .iter()
                .last()
                .unwrap()
                .1
                .iter()
                .min_by(|&x, &y| dist[v_i][v_j][x.0][x.1].cmp(&dist[v_i][v_j][y.0][y.1]))
                .unwrap();
        }
    }

    "".to_owned()
}

/// 貪欲法
/// 現在の間隔をt、理想的な間隔をt'、距離をdとして、a^(t+d-t')が大きいところを掃除しに行く
/// タイブレークも同じ指標を使用
pub fn solve(input: &Input) -> String {
    let (mut s_i, mut s_j) = (0, 0);
    let adj = create_adj(input);
    let r = calc_r(input);
    const ALPHA: f64 = 1.01;

    let mut ans = vec![];
    let mut prev = vec![vec![-1; input.n]; input.n];
    prev[s_i][s_j] = 0;

    loop {
        let dist = calc_dist((s_i, s_j), input, &adj);
        let mut eval = vec![vec![0.; input.n]; input.n];
        let (mut best_i, mut best_j) = (0, 0);

        for i in 0..input.n {
            for j in 0..input.n {
                if s_i == i && s_j == j {
                    continue;
                }
                let t = ans.len() as i64 - prev[i][j];
                let ideal_t = 1. / r[i][j];
                eval[i][j] =
                    input.d[i][j] as f64 * ALPHA.powf((t - 10 * dist[i][j]) as f64 / ideal_t);

                if eval[i][j] > eval[best_i][best_j] {
                    (best_i, best_j) = (i, j);
                }
            }
        }

        let cand = get_move_cand((s_i, s_j), (best_i, best_j), &dist, &adj);
        let best_d = cand
            .iter()
            .copied()
            .max_by(|d1, d2| {
                let v1 = d1.add((s_i, s_j));
                let v2 = d2.add((s_i, s_j));
                eval[v2.0][v2.1].partial_cmp(&eval[v1.0][v1.1]).unwrap()
            })
            .unwrap();
        ans.push(best_d);
        (s_i, s_j) = best_d.add((s_i, s_j));
        prev[s_i][s_j] = ans.len() as i64;

        // 終了条件
        if (s_i, s_j) == (0, 0) {
            let mut reached_count = 0;
            for i in 0..input.n {
                for j in 0..input.n {
                    if prev[i][j] >= 0 {
                        reached_count += 1;
                    }
                }
            }
            if reached_count == input.n * input.n {
                break;
            }
        }
    }

    ans.iter().map(|x| x.to_char()).collect()
}
