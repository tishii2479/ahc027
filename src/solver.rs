use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::solver_util::*;
use crate::util::*;

const TSP_ITER_CNT: usize = 100000;
const TOTAL_LENGTH: usize = 1e4 as usize;
const INF: i64 = 1e17 as i64;
const ALPHA: f64 = 0.8;

pub fn solve(input: &Input) -> Vec<(usize, usize)> {
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);
            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    let s = s;
    let mut total_length = 0;
    let ideal_cycle_l = (2.0 / r[s.0][s.1]).round() as usize;
    let mut cycles = vec![];
    let mut counts = vec![vec![0; input.n]; input.n];
    let pre_p = -(input.n.pow(2) as f64 / TOTAL_LENGTH as f64);
    let mut ps: Vec<Vec<BTreeSet<FloatIndex>>> =
        vec![vec![BTreeSet::from([FloatIndex(pre_p)]); input.n]; input.n];

    let gain_size: usize = input.n * input.n / 12;

    // サイクルの作成
    while total_length < TOTAL_LENGTH as i64 {
        let mut gain_cand = vec![];
        let start_t = total_length as f64 / TOTAL_LENGTH as f64;
        let end_t = (total_length + ideal_cycle_l as i64) as f64 / TOTAL_LENGTH as f64;
        for i in 0..input.n {
            for j in 0..input.n {
                gain_cand.push((
                    calc_prev_delta(FloatIndex((start_t + end_t) / 2.), &ps[i][j])
                        * input.d[i][j] as f64,
                    (i, j),
                ));
            }
        }
        gain_cand.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut selected_v = vec![s];
        for i in 0..gain_size {
            selected_v.push(gain_cand[i].1);
        }
        let (cycle, _) =
            create_single_cycle(&selected_v, start_t, end_t, &dist, &mut ps, input, &adj);

        eprint!("{} ", cycle.len());
        for v in cycle.iter() {
            counts[v.0][v.1] += 1;
        }

        total_length += cycle.len() as i64;
        cycles.push(cycle);
    }
    eprintln!();

    // 回収されなかったところを全て回収するサイクルを作る
    let mut unvisited = vec![];
    for i in 0..input.n {
        for j in 0..input.n {
            if counts[i][j] == 0 {
                unvisited.push((i, j));
            }
        }
    }
    if unvisited.len() > 0 {
        eprintln!("unvisited counts:    {}", unvisited.len());
        unvisited.insert(0, s);
        cycles.push(
            create_single_cycle(
                &unvisited,
                total_length as f64 / TOTAL_LENGTH as f64,
                (total_length + ideal_cycle_l as i64) as f64 / TOTAL_LENGTH as f64,
                &dist,
                &mut ps,
                input,
                &adj,
            )
            .0,
        );
    }

    let cycle_cnt = cycles.len();

    eprintln!("s:               {:?}", s);
    eprintln!("ideal_cycle_l:   {ideal_cycle_l}");
    eprintln!("cycle_cnt:       {cycle_cnt}");
    eprintln!("total_length:    {}", total_length);
    eprintln!(
        "average_length:  {:.3}",
        total_length as f64 / cycle_cnt as f64
    );

    // let state = optimize_cycles(cycles, s, &dist, &adj, input);
    // let mut cycle_order = vec![0; cycle_cnt];
    // for (i, status) in state.cycle_usage.iter().enumerate() {
    //     if *status == UNUSED {
    //         continue;
    //     }
    //     cycle_order[*status] = i;
    // }
    // let cycles = cycle_order.iter().map(|&i| state.cycles[i].clone()).collect();
    let cycle_order: Vec<usize> = (0..cycle_cnt).collect();
    let cycles = cycle_order.iter().map(|&i| cycles[i].clone()).collect();

    let path = cycles_to_path(&cycles);

    path
}

fn create_single_cycle(
    v: &Vec<(usize, usize)>,
    start_t: f64,
    end_t: f64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    ps: &mut Vec<Vec<BTreeSet<FloatIndex>>>,
    input: &Input,
    adj: &Adj,
) -> (Vec<(usize, usize)>, f64) {
    let (order, dist_sum) = solve_tsp(&v, dist, TSP_ITER_CNT);
    let p = order.iter().position(|x| x == &0).unwrap();
    let order: Vec<(usize, usize)> = order.iter().map(|&i| v[(i + p) % v.len()]).collect();
    let mut cycle = vec![];
    let mut score_delta = 0.;
    for p_i in 0..order.len() {
        let path = find_best_path(
            order[p_i],
            order[(p_i + 1) % order.len()],
            start_t,
            end_t,
            dist,
            ps,
            input,
            adj,
        );

        for (i, v) in path.iter().enumerate() {
            // 最初に仮の値を入れているので、それを削除する
            if ps[v.0][v.1].len() == 1 && ps[v.0][v.1].iter().next().unwrap() < &FloatIndex(0.) {
                ps[v.0][v.1].clear();
            }

            // ISSUE: `i + cycle.len()`が正しいが、 `i`の方がスコアが良い、謎
            // おそらく、厳しく評価した方が良いので、i=0の時によくなりがち
            let index = to_float_index(
                start_t,
                start_t * ALPHA + end_t * (1. - ALPHA),
                i + cycle.len(),
                dist_sum as usize,
            );
            ps[v.0][v.1].insert(index);
            score_delta += calc_delta(index, &ps[v.0][v.1]) * input.d[v.0][v.1] as f64;
        }
        cycle.extend(path);
    }
    (cycle, score_delta)
}

fn find_best_path(
    from: (usize, usize),
    to: (usize, usize),
    start_t: f64,
    end_t: f64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    ps: &mut Vec<Vec<BTreeSet<FloatIndex>>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let mut dp = vec![vec![-INF as f64; input.n]; input.n];
    let mut q = VecDeque::new();
    q.push_back((from, 0.));
    dp[from.0][from.1] = 0.;

    while let Some((v, val)) = q.pop_front() {
        if dp[v.0][v.1] > val {
            continue;
        }
        for (_, u) in adj[v.0][v.1].iter() {
            let is_closer = dist[to.0][to.1][u.0][u.1] < dist[to.0][to.1][v.0][v.1];
            if !is_closer {
                continue;
            }
            let gain = calc_prev_delta(FloatIndex((start_t + end_t) / 2.), &ps[u.0][u.1])
                * input.d[u.0][u.1] as f64;
            let new_val = dp[v.0][v.1] + gain;
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
        let gain = calc_prev_delta(FloatIndex((start_t + end_t) / 2.), &ps[cur.0][cur.1])
            * input.d[cur.0][cur.1] as f64;
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

fn to_float_index(start_t: f64, end_t: f64, i: usize, cycle_length: usize) -> FloatIndex {
    FloatIndex(start_t + (end_t - start_t) * i as f64 / cycle_length as f64)
}
