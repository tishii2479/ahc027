use std::collections::VecDeque;

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
    let cycle_l = (1. / r[s.0][s.1]).round() as i64;
    let cycle_cnt = L / cycle_l as usize;
    let mut cycles = vec![];

    // サイクルの作成
    for _ in 0..cycle_cnt {
        let cycle = create_cycle(
            s,
            cycle_l,
            &dist,
            &required_counts,
            &mut counts,
            input,
            &adj,
        );
        // show(&counts, &required_count);
        // show_path(&cycle, input.n);
        cycles.push(cycle);
    }

    show(&counts, &required_counts, input);
    rnd::shuffle(&mut cycles);

    let ans = cycles_to_answer(&cycles);

    eprintln!("s:               {:?}", s);
    eprintln!("cycle_l:         {cycle_l}");
    eprintln!("cycle_cnt:       {cycle_cnt}");
    eprintln!("total_length:    {}", ans.len());

    ans
}

fn solve_tsp(v: &Vec<(usize, usize)>, dist: &Vec<Vec<Vec<Vec<i64>>>>) -> (Vec<usize>, i64) {
    const LOOP_COUNT: usize = 1000;
    let n = v.len();
    let mut order: Vec<usize> = (0..n).map(|x| x).collect();
    let mut score = 0;
    for i in 0..n {
        let j = (i + 1) % n;
        score += dist[v[i].0][v[i].1][v[j].0][v[j].1];
    }

    for _ in 0..=LOOP_COUNT {
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
            // 採用
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
    }

    (order, score)
}

fn find_best_path(
    s: (usize, usize),
    t: (usize, usize),
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    counts: &Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let mut dp = vec![vec![-INF as f64; input.n]; input.n];
    let mut q = VecDeque::new();
    q.push_back((s, 0.));
    dp[s.0][s.1] = 0.;

    while let Some((v, val)) = q.pop_front() {
        if dp[v.0][v.1] > val {
            continue;
        }
        for (_, u) in adj[v.0][v.1].iter() {
            let is_closer = dist[t.0][t.1][u.0][u.1] < dist[t.0][t.1][v.0][v.1];
            if !is_closer {
                continue;
            }
            let new_val = dp[v.0][v.1] + calc_gain(counts[u.0][u.1], input.d[u.0][u.1]);
            if new_val > dp[u.0][u.1] {
                dp[u.0][u.1] = new_val;
                q.push_back((*u, dp[u.0][u.1]));
            }
        }
    }

    // 復元
    let mut path = vec![];
    let mut cur = t;
    while cur != s {
        let gain = calc_gain(counts[cur.0][cur.1], input.d[cur.0][cur.1]);
        let (_, nxt) = adj[cur.0][cur.1]
            .iter()
            .filter(|(_, u)| dist[s.0][s.1][u.0][u.1] < dist[s.0][s.1][cur.0][cur.1])
            .min_by(|(_, v), (_, u)| {
                (dp[cur.0][cur.1] - dp[v.0][v.1] - gain)
                    .abs()
                    .partial_cmp(&(dp[cur.0][cur.1] - dp[u.0][u.1] - gain).abs())
                    .unwrap()
            })
            .unwrap();
        path.push(cur);
        cur = *nxt;
    }
    path.reverse();
    path
}

fn create_cycle(
    s: (usize, usize),
    cycle_l: i64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    required_counts: &Vec<Vec<i64>>,
    counts: &mut Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    const COUNT_SIZE: usize = 20;
    const GAIN_SIZE: usize = 5;

    let mut gain_cand = vec![];
    let mut count_cand = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            gain_cand.push((calc_gain(counts[i][j], input.d[i][j]), (i, j)));
            count_cand.push((required_counts[i][j] - counts[i][j], (i, j)));
        }
    }
    let mut selected_v = vec![s];

    gain_cand.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for i in 0..GAIN_SIZE {
        selected_v.push(gain_cand[i].1);
    }
    count_cand.sort_by(|a, b| b.cmp(a));
    for i in 0..COUNT_SIZE {
        if selected_v.contains(&count_cand[i].1) {
            continue;
        }
        selected_v.push(count_cand[i].1);
    }

    let mut rev_counts = vec![];
    let (order, dist_sum) = solve_tsp(&selected_v, dist);
    let p = order.iter().position(|x| x == &0).unwrap();
    let order: Vec<(usize, usize)> = order
        .iter()
        .map(|&i| selected_v[(i + p) % selected_v.len()])
        .collect();
    let mut cycle = vec![];
    for i in 0..order.len() {
        let path = find_best_path(
            order[i],
            order[(i + 1) % order.len()],
            dist,
            counts,
            input,
            adj,
        );

        for v in path.iter() {
            if counts[v.0][v.1] == USED_CODE {
                continue;
            }
            rev_counts.push((*v, counts[v.0][v.1] + 1));
            counts[v.0][v.1] = USED_CODE;
        }
        cycle.extend(path);
    }

    for (v, rev_t) in rev_counts {
        counts[v.0][v.1] = rev_t;
    }

    show_path(&cycle, input.n);
    eprintln!("{}", cycle.len());
    cycle
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

fn calc_gain(count: i64, d: i64) -> f64 {
    (d as f64 * (1. / (count as f64 + EPS).powf(2.) - 1. / (count as f64 + 1.).powf(2.)))
        .min(EDGE_WEIGHT as f64 - 1.)
}

#[allow(unused)]
fn show(counts: &Vec<Vec<i64>>, required_counts: &Vec<Vec<i64>>, input: &Input) {
    eprintln!("-----");
    let mut lower_bound = 0.;
    let mut sum = 0;
    let mut max = 0;
    for i in 0..counts.len() {
        for j in 0..counts[i].len() {
            lower_bound +=
                input.d[i][j] as f64 * (L as f64 / (counts[i][j] as f64 + 1e-6)).powf(2.);
            eprint!("{:4}", required_counts[i][j] - counts[i][j]);
            if required_counts[i][j] - counts[i][j] > 0 {
                max = max.max(required_counts[i][j] - counts[i][j]);
                sum += required_counts[i][j] - counts[i][j];
            }
        }
        eprintln!();
    }
    eprintln!("lower:   {:.5}", lower_bound / L as f64);
    eprintln!("max:     {max}");
    eprintln!("ave:     {:.5}", sum as f64 / counts.len().pow(2) as f64);
}

#[allow(unused)]
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
