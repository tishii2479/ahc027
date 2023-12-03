use std::collections::{BinaryHeap, VecDeque};

use crate::def::*;
use crate::util::*;

type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

const L: usize = 1e4 as usize;
const INF: i64 = 1e17 as i64;
const EPS: f64 = 1e-6;
const EDGE_WEIGHT: i64 = 1e9 as i64; // NOTE: USED_CODEより小さくあるべき
const USED_CODE: i64 = 1e9 as i64;

pub fn solve(input: &Input) -> String {
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);
    let mut counts = vec![vec![0; input.n]; input.n];
    let mut required_counts = vec![vec![0; input.n]; input.n];

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);

            required_counts[i][j] = (L as f64 * r[i][j]).round() as i64;

            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    show(&counts, &required_counts, input);

    let s = s;
    let cycle_l = (1.2 / r[s.0][s.1]).round() as i64;
    let cycle_cnt = L / cycle_l as usize;
    let mut cycles = vec![];

    // サイクルの作成
    for _ in 0..cycle_cnt {
        let cycle = create_cycle(s, cycle_l, &dist, &mut counts, input, &adj);
        // show(&counts, &required_count);
        show_path(&cycle, input.n);
        cycles.push(cycle);
    }

    show(&counts, &required_counts, input);
    rnd::shuffle(&mut cycles);

    eprintln!("s:           {:?}", s);
    eprintln!("cycle_l:     {cycle_l}");
    eprintln!("cycle_cnt:   {cycle_cnt}");

    cycles_to_answer(&cycles)
}

fn show_path(path: &Vec<(usize, usize)>, n: usize) {
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

fn calc_gain(count: i64, d: i64) -> f64 {
    (d as f64 * (1. / (count as f64 + EPS).powf(2.) - 1. / (count as f64 + 1.).powf(2.)))
        .min(EDGE_WEIGHT as f64 - 1.)
}

fn create_cycle(
    s: (usize, usize),
    cycle_l: i64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    counts: &mut Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let mut v = (s.0, s.1);
    let mut path = vec![];
    let mut rev_counts = vec![];

    while cycle_l - path.len() as i64 > dist[s.0][s.1][v.0][v.1] {
        let mut last = vec![];
        let mut eval = vec![vec![0.; input.n]; input.n];
        let mut cands = vec![];
        for i in 0..input.n {
            for j in 0..input.n {
                eval[i][j] = calc_gain(counts[i][j], input.d[i][j]);
                cands.push((eval[i][j], (i, j)));
            }
        }
        cands.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let cand_size = if path.len() == 0 {
            2
        } else {
            input.n * input.n / 16
        };
        for i in 0..cand_size {
            last.push(cands[i].1);
        }

        assert!(last.len() > 0);
        let target_v = last
            .iter()
            .filter(|&x| x != &v)
            .min_by(|&x, &y| dist[v.0][v.1][x.0][x.1].cmp(&dist[v.0][v.1][y.0][y.1]))
            .unwrap();

        let add_path = shortest_path(&v, target_v, &counts, input, &adj);
        for v in add_path.iter() {
            if counts[v.0][v.1] == USED_CODE {
                continue;
            }
            rev_counts.push((*v, counts[v.0][v.1] + 1));
            counts[v.0][v.1] = USED_CODE;
        }
        path.extend(add_path);
        v = *target_v;
    }

    // vからsに戻る
    let return_path = shortest_path(&v, &s, &counts, input, &adj);
    for v in return_path.iter() {
        if counts[v.0][v.1] == USED_CODE {
            continue;
        }
        rev_counts.push((*v, counts[v.0][v.1] + 1));
    }
    path.extend(return_path);

    for (v, rev_t) in rev_counts {
        counts[v.0][v.1] = rev_t;
    }

    path
}

fn show(counts: &Vec<Vec<i64>>, required_count: &Vec<Vec<i64>>, input: &Input) {
    eprintln!("-----");
    let mut lower_bound = 0.;
    let mut sum = 0;
    let mut max = 0;
    for i in 0..counts.len() {
        for j in 0..counts[i].len() {
            lower_bound +=
                input.d[i][j] as f64 * (L as f64 / (counts[i][j] as f64 + 1e-6)).powf(2.);
            eprint!("{:4}", required_count[i][j] - counts[i][j]);
            if required_count[i][j] - counts[i][j] > 0 {
                max = max.max(required_count[i][j] - counts[i][j]);
                sum += required_count[i][j] - counts[i][j];
            }
        }
        eprintln!();
    }
    eprintln!("lower:   {:.5}", lower_bound / L as f64);
    eprintln!("max:     {max}");
    eprintln!("ave:     {:.5}", sum as f64 / counts.len().pow(2) as f64);
}

fn shortest_path(
    s: &(usize, usize),
    t: &(usize, usize),
    counts: &Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    use std::cmp::Reverse;
    let mut dist = vec![vec![INF; input.n]; input.n];
    let mut q = BinaryHeap::new();
    q.push((Reverse(0), s));
    dist[s.0][s.1] = 0;
    while let Some((Reverse(d), v)) = q.pop() {
        if v == t {
            // tまで探索が終わっていたら打ち切る
            break;
        }
        if d > dist[v.0][v.1] {
            continue;
        }
        for (_, u) in adj[v.0][v.1].iter() {
            let cost = EDGE_WEIGHT - calc_gain(counts[u.0][u.1], input.d[u.0][u.1]) as i64;
            if dist[u.0][u.1] <= d + cost {
                continue;
            }
            dist[u.0][u.1] = d + cost;
            q.push((Reverse(dist[u.0][u.1]), u));
        }
    }

    // 復元
    let mut path = vec![];
    let mut cur = *t;
    while cur != *s {
        for (_, u) in adj[cur.0][cur.1].iter() {
            let cost = EDGE_WEIGHT - calc_gain(counts[cur.0][cur.1], input.d[cur.0][cur.1]) as i64;
            if dist[cur.0][cur.1] == dist[u.0][u.1] + cost {
                path.push(cur);
                cur = *u;
                break;
            }
        }
    }
    path.reverse();
    path
}

fn cycles_to_answer(cycles: &Vec<Vec<(usize, usize)>>) -> String {
    let mut path = vec![];
    for cycle in cycles {
        path.extend(cycle.clone());
    }
    let start = path
        .iter()
        .position(|&v| v == (0, 0))
        .expect("No (0, 0) in cycle");

    let mut ans = vec![];
    for i in start..start + path.len() {
        ans.push(Dir::from(path[i % path.len()], path[(i + 1) % path.len()]));
    }
    ans.iter().map(|d| d.to_char()).collect()
}

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
