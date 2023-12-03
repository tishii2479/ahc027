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
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    adj: &Adj,
) -> Vec<Dir> {
    let mut v = vec![];
    for (dir, nxt) in adj[s.0][s.1].iter() {
        if dist[t.0][t.1][s.0][s.1] == dist[t.0][t.1][nxt.0][nxt.1] + 1 {
            v.push(dir.to_owned());
        }
    }
    v
}

pub fn solve(input: &Input) -> String {
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);
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

            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    let s = s;
    let cycle_l = (1. / r[s.0][s.1]).round() as usize;
    let cycle_cnt = (1e5 as usize) / cycle_l;
    let mut cycles = vec![];

    for _ in 0..cycle_cnt {
        let mut v = (s.0, s.1);
        let mut path = vec![];
        while cycle_l - path.len() > dist[s.0][s.1][v.0][v.1] as usize {
            let target_v = counts
                .iter()
                .last()
                .unwrap()
                .1
                .iter()
                .min_by(|&x, &y| dist[v.0][v.1][x.0][x.1].cmp(&dist[v.0][v.1][y.0][y.1]))
                .unwrap();

            let next_dir = get_move_cand(v, *target_v, &dist, &adj)
                .first()
                .unwrap()
                .to_owned();
            v = next_dir.add(v);
            path.push(v);
        }

        while v != s {
            let next_dir = get_move_cand(v, s, &dist, &adj).first().unwrap().to_owned();
            v = next_dir.add(v);
            path.push(v);
        }

        eprintln!("{:?}", path);
        cycles.push(path);
    }
    panic!();

    "".to_owned()
}
