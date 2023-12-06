use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::util::*;

type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

const TOTAL_LENGTH: usize = 1e4 as usize;
const INF: i64 = 1e17 as i64;

pub fn solve(input: &Input) -> Vec<(usize, usize)> {
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);
    let mut required_counts = vec![vec![0; input.n]; input.n];
    let mut prev = vec![vec![0; input.n]; input.n];

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);
            required_counts[i][j] = (TOTAL_LENGTH as f64 * r[i][j]).round() as i64;

            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    // show(&counts, &required_counts, input);

    let s = s;
    let mut t = 0;
    let ideal_cycle_l = (1.2 / r[s.0][s.1]).round() as usize;
    let cycle_cnt = TOTAL_LENGTH / ideal_cycle_l;
    let mut cycles = vec![];

    // サイクルの作成
    for _ in 0..cycle_cnt {
        let cycle = create_cycle(s, t, ideal_cycle_l, &dist, &mut prev, input, &adj);
        t += cycle.len() as i64;
        cycles.push(cycle);
    }

    // show(&counts, &required_counts, input);

    // rnd::shuffle(&mut cycles);
    // let use_cycles = optimize_cycles(cycle_cnt, ideal_cycle_l, &cycles, input);
    let use_cycles: Vec<usize> = (0..cycle_cnt).collect();
    let cycles = use_cycles.iter().map(|&i| cycles[i].clone()).collect();

    let path = cycles_to_path(&cycles);

    eprintln!("s:               {:?}", s);
    eprintln!("ideal_cycle_l:   {ideal_cycle_l}");
    eprintln!("cycle_cnt:       {cycle_cnt}");
    eprintln!("total_length:    {}", path.len());

    path
}

fn create_cycle(
    s: (usize, usize),
    t: i64,
    ideal_cycle_l: usize,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    prev: &mut Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let gain_size: usize = input.n * input.n / 8;

    let mut gain_cand = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            gain_cand.push((calc_gain2(t, prev[i][j], input.d[i][j]), (i, j)));
        }
    }
    let mut selected_v = vec![s];

    gain_cand.sort_by(|a, b| b.partial_cmp(a).unwrap());
    for i in 0..gain_size {
        selected_v.push(gain_cand[i].1);
    }

    let (order, _) = solve_tsp(&selected_v, dist);
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
            t + cycle.len() as i64,
            dist,
            prev,
            input,
            adj,
        );

        for (i, v) in path.iter().enumerate() {
            prev[v.0][v.1] = t + i as i64;
        }
        cycle.extend(path);
    }

    // show_path(&cycle, input.n);
    // eprintln!("{}", cycle.len());
    cycle
}

fn find_best_path(
    from: (usize, usize),
    to: (usize, usize),
    t: i64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    prev: &Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let mut dp = vec![vec![-INF; input.n]; input.n];
    let mut q = VecDeque::new();
    q.push_back((from, 0));
    dp[from.0][from.1] = 0;

    while let Some((v, val)) = q.pop_front() {
        if dp[v.0][v.1] > val {
            continue;
        }
        for (_, u) in adj[v.0][v.1].iter() {
            let is_closer = dist[to.0][to.1][u.0][u.1] < dist[to.0][to.1][v.0][v.1];
            if !is_closer {
                continue;
            }
            let new_val = dp[v.0][v.1] + calc_gain2(t, prev[u.0][u.1], input.d[u.0][u.1]);
            if new_val > dp[u.0][u.1] {
                dp[u.0][u.1] = new_val;
                q.push_back((*u, dp[u.0][u.1]));
            }
        }
    }

    // 復元
    let mut path = vec![];
    let mut cur = to;
    while cur != from {
        let gain = calc_gain2(t, prev[cur.0][cur.1], input.d[cur.0][cur.1]);
        let (_, nxt) = adj[cur.0][cur.1]
            .iter()
            .filter(|(_, u)| dist[from.0][from.1][u.0][u.1] < dist[from.0][from.1][cur.0][cur.1])
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

fn optimize_cycles(
    cycle_cnt: usize,
    ideal_cycle_l: usize,
    cycles: &Vec<Vec<(usize, usize)>>,
    input: &Input,
) -> Vec<usize> {
    const UNUSED: usize = usize::MAX;
    let total_cycle_length = (cycle_cnt * ideal_cycle_l) as i64;
    let mut cycle_status: Vec<usize> = (0..cycle_cnt).collect();
    cycle_status.extend(vec![UNUSED; cycles.len()]);

    let mut ps: Vec<Vec<BTreeSet<FloatIndex>>> = vec![vec![BTreeSet::new(); input.n]; input.n];

    fn to_float_index(t: usize, i: usize, cycle_l: usize, ideal_cycle_l: usize) -> FloatIndex {
        FloatIndex((t * ideal_cycle_l) as f64 + ideal_cycle_l as f64 * i as f64 / cycle_l as f64)
    }

    fn action_swap(
        c_a: usize,
        c_b: usize,
        cycle_status: &mut Vec<usize>,
        cycles: &Vec<Vec<(usize, usize)>>,
        ps: &mut Vec<Vec<BTreeSet<FloatIndex>>>,
        total_cycle_length: i64,
        ideal_cycle_l: usize,
    ) -> f64 {
        let mut score_delta = 0.;
        // 取り除く
        for c_i in [c_a, c_b] {
            if cycle_status[c_i] == UNUSED {
                continue;
            }
            let t = cycle_status[c_i];
            for (i, v) in cycles[c_i].iter().enumerate() {
                let index = to_float_index(t, i, cycles[c_i].len(), ideal_cycle_l);
                score_delta -= calc_delta(index, &ps[v.0][v.1], total_cycle_length, true);
                ps[v.0][v.1].remove(&index);
            }
        }
        cycle_status.swap(c_a, c_b);

        // 追加する
        for c_i in [c_a, c_b] {
            if cycle_status[c_i] == UNUSED {
                continue;
            }
            let t = cycle_status[c_i];
            for (i, v) in cycles[c_i].iter().enumerate() {
                let index = to_float_index(t, i, cycles[c_i].len(), ideal_cycle_l);
                ps[v.0][v.1].insert(index);
                score_delta += calc_delta(index, &ps[v.0][v.1], total_cycle_length, true);
            }
        }

        score_delta
    }

    for c_i in 0..cycle_cnt {
        for (i, v) in cycles[c_i].iter().enumerate() {
            ps[v.0][v.1].insert(to_float_index(c_i, i, cycles[c_i].len(), ideal_cycle_l));
        }
    }

    let mut score = 0.;
    for i in 0..input.n {
        for j in 0..input.n {
            for x in ps[i][j].iter() {
                score += calc_delta(*x, &ps[i][j], total_cycle_length, false);
            }
        }
    }

    let mut iter_count = 0;
    while time::elapsed_seconds() < TIME_LIMIT {
        let (c_a, c_b) = (
            rnd::gen_range(0, cycles.len()),
            rnd::gen_range(0, cycles.len()),
        );
        if c_a == c_b || (cycle_status[c_a] == UNUSED && cycle_status[c_b] == UNUSED) {
            continue;
        }
        iter_count += 1;

        let prev_score = score;

        score += action_swap(
            c_a,
            c_b,
            &mut cycle_status,
            cycles,
            &mut ps,
            total_cycle_length,
            ideal_cycle_l,
        );

        if score < prev_score {
            // eprintln!("adopt: {prev_score} -> {score} {c_a} {c_b}");
        } else {
            score += action_swap(
                c_a,
                c_b,
                &mut cycle_status,
                cycles,
                &mut ps,
                total_cycle_length,
                ideal_cycle_l,
            );
        }
    }

    eprintln!("iter_count: {}", iter_count);

    let mut cycle_order = vec![0; cycle_cnt];
    for (i, status) in cycle_status.iter().enumerate() {
        if *status == UNUSED {
            continue;
        }
        cycle_order[*status] = i;
    }
    cycle_order
}

fn solve_tsp(v: &Vec<(usize, usize)>, dist: &Vec<Vec<Vec<Vec<i64>>>>) -> (Vec<usize>, i64) {
    const LOOP_COUNT: usize = 10000;
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

fn calc_delta(
    x: FloatIndex,
    btree: &BTreeSet<FloatIndex>,
    total_cycle_length: i64,
    delta: bool,
) -> f64 {
    let first = btree.iter().next().unwrap().0 + total_cycle_length as f64;
    let last = btree.iter().next_back().unwrap().0 - total_cycle_length as f64;
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
    ((((x.0 - prev) as f64).powf(2.) + ((next - x.0) as f64).powf(2.))
        - if delta {
            ((next - prev) as f64).powf(2.)
        } else {
            0.
        })
        * if delta { 2. } else { 1. }
}

fn cycles_to_path(cycles: &Vec<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
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

fn calc_gain2(t: i64, prev: i64, d: i64) -> i64 {
    (t - prev).pow(2) * d
}

#[allow(unused)]
fn show(counts: &Vec<Vec<i64>>, required_counts: &Vec<Vec<i64>>, input: &Input) {
    eprintln!("-----");
    let mut lower_bound = 0.;
    let mut sum = 0;
    let mut max = 0;
    for i in 0..counts.len() {
        for j in 0..counts[i].len() {
            lower_bound += input.d[i][j] as f64
                * (TOTAL_LENGTH as f64 / (counts[i][j] as f64 + 1e-6)).powf(2.);
            eprint!("{:4}", required_counts[i][j] - counts[i][j]);
            if required_counts[i][j] - counts[i][j] > 0 {
                max = max.max(required_counts[i][j] - counts[i][j]);
                sum += required_counts[i][j] - counts[i][j];
            }
        }
        eprintln!();
    }
    eprintln!("lower:   {:.5}", lower_bound / TOTAL_LENGTH as f64);
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

#[test]
fn test_calc_delta() {
    let mut btree = BTreeSet::new();
    let total_cycle_length = 1000000;
    for _ in 0..10 {
        btree.insert(FloatIndex(rnd::gen_range(0, total_cycle_length) as f64));
    }
    fn calc_actual_score(btree: &BTreeSet<FloatIndex>, total_cycle_length: i64) -> f64 {
        let mut score = 0.;
        for x in btree.iter() {
            score += calc_delta(*x, &btree, total_cycle_length, false);
        }
        score
    }
    let mut score = calc_actual_score(&btree, total_cycle_length as i64);
    for _ in 0..10000 {
        let prev_score = score;
        let x = FloatIndex(rnd::gen_range(0, total_cycle_length) as f64);
        if btree.contains(&x) {
            continue;
        }

        btree.insert(x);
        score += calc_delta(x, &btree, total_cycle_length as i64, true);
        assert_eq!(score, calc_actual_score(&btree, total_cycle_length as i64));

        score -= calc_delta(x, &btree, total_cycle_length as i64, true);
        btree.remove(&x);
        assert_eq!(prev_score, score);
    }
}
