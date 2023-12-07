use std::collections::BTreeSet;
use std::collections::VecDeque;

use crate::def::*;
use crate::solver_util::*;
use crate::util::*;

const TSP_ITER_CNT: usize = 100000;
const TOTAL_LENGTH: usize = 1e4 as usize;
const INF: i64 = 1e17 as i64;
const UNUSED: usize = usize::MAX;

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
    let ideal_cycle_l = (1.1 / r[s.0][s.1]).round() as usize;
    let mut cycles = vec![];
    let mut counts = vec![vec![0; input.n]; input.n];
    let pre_p = -(input.n.pow(2) as f64 / TOTAL_LENGTH as f64);
    let mut ps: Vec<Vec<BTreeSet<FloatIndex>>> =
        vec![vec![BTreeSet::from([FloatIndex(pre_p)]); input.n]; input.n];

    // サイクルの作成
    while total_length < TOTAL_LENGTH as i64 {
        let cycle = create_cycle(
            s,
            total_length as f64 / TOTAL_LENGTH as f64,
            (total_length + ideal_cycle_l as i64) as f64 / TOTAL_LENGTH as f64,
            &dist,
            &mut ps,
            input,
            &adj,
        );
        total_length += cycle.len() as i64;
        eprint!("{} ", cycle.len());
        for v in cycle.iter() {
            counts[v.0][v.1] += 1;
        }
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
        cycles.push(create_single_cycle(
            &unvisited,
            total_length as f64 / TOTAL_LENGTH as f64,
            (total_length + ideal_cycle_l as i64) as f64 / TOTAL_LENGTH as f64,
            &dist,
            &mut ps,
            input,
            &adj,
        ));
    }

    show(&cycles, input);
    let cycle_cnt = cycles.len();

    eprintln!("s:               {:?}", s);
    eprintln!("ideal_cycle_l:   {ideal_cycle_l}");
    eprintln!("cycle_cnt:       {cycle_cnt}");
    eprintln!("total_length:    {}", total_length);

    // let use_cycles = optimize_cycles(cycle_cnt, ideal_cycle_l, &cycles, input);
    let use_cycles: Vec<usize> = (0..cycle_cnt).collect();
    let cycles = use_cycles.iter().map(|&i| cycles[i].clone()).collect();

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
) -> Vec<(usize, usize)> {
    let (order, dist_sum) = solve_tsp(&v, dist, TSP_ITER_CNT);
    let p = order.iter().position(|x| x == &0).unwrap();
    let order: Vec<(usize, usize)> = order.iter().map(|&i| v[(i + p) % v.len()]).collect();
    let mut cycle = vec![];
    for i in 0..order.len() {
        let path = find_best_path(
            order[i],
            order[(i + 1) % order.len()],
            start_t,
            end_t,
            dist,
            ps,
            input,
            adj,
        );

        for (i, v) in path.iter().enumerate() {
            if ps[v.0][v.1].len() == 1 && ps[v.0][v.1].iter().next().unwrap() < &FloatIndex(0.) {
                ps[v.0][v.1].clear();
            }
            ps[v.0][v.1].insert(FloatIndex(
                start_t + (end_t - start_t) * i as f64 / dist_sum as f64,
            ));
        }
        cycle.extend(path);
    }
    cycle
}

fn create_cycle(
    s: (usize, usize),
    start_t: f64,
    end_t: f64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    ps: &mut Vec<Vec<BTreeSet<FloatIndex>>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let gain_size: usize = input.n * input.n / 12;
    let mut gain_cand = vec![];
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

    create_single_cycle(&selected_v, start_t, end_t, dist, ps, input, adj)
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

fn to_float_index(t: usize, i: usize, cycle_l: usize) -> FloatIndex {
    FloatIndex(t as f64 + i as f64 / cycle_l as f64)
}

struct State {
    score: f64,
    total_cycle_length: i64,
    cycles: Vec<Vec<(usize, usize)>>,
    cycle_usage: Vec<usize>,
    ps: Vec<Vec<BTreeSet<FloatIndex>>>,
}

impl State {
    fn new(cycles: Vec<Vec<(usize, usize)>>, total_cycle_length: i64, input: &Input) -> State {
        let cycle_cnt = cycles.len();
        let cycle_usage = (0..cycle_cnt).collect();
        let mut ps: Vec<Vec<BTreeSet<FloatIndex>>> = vec![vec![BTreeSet::new(); input.n]; input.n];

        for c_i in 0..cycle_cnt {
            for (i, v) in cycles[c_i].iter().enumerate() {
                ps[v.0][v.1].insert(to_float_index(c_i, i, cycles[c_i].len()));
            }
        }

        let mut score = 0.;
        for i in 0..input.n {
            for j in 0..input.n {
                for x in ps[i][j].iter() {
                    let (prev, next) = get_prev_and_next(*x, &ps[i][j]);
                    score += ((prev - x.0).powf(2.) + (next - x.0).powf(2.)) / 2.;
                    score += calc_delta(*x, &ps[i][j]) * input.d[i][j] as f64;
                }
            }
        }

        State {
            score,
            total_cycle_length,
            cycles,
            cycle_usage,
            ps,
        }
    }

    fn remove_cycle(&mut self, c_i: usize, input: &Input) -> f64 {
        let mut score_delta = 0.;
        let t = self.cycle_usage[c_i];
        assert_ne!(t, UNUSED);
        for (i, v) in self.cycles[c_i].iter().enumerate() {
            let index = to_float_index(t, i, self.cycles[c_i].len());
            score_delta -= calc_delta(index, &self.ps[v.0][v.1]) * input.d[v.0][v.1] as f64;
            self.ps[v.0][v.1].remove(&index);
        }
        score_delta
    }

    fn insert_cycle(&mut self, c_i: usize, input: &Input) -> f64 {
        let mut score_delta = 0.;
        let t = self.cycle_usage[c_i];
        assert_ne!(t, UNUSED);
        for (i, v) in self.cycles[c_i].iter().enumerate() {
            let index = to_float_index(t, i, self.cycles[c_i].len());
            self.ps[v.0][v.1].insert(index);
            score_delta += calc_delta(index, &self.ps[v.0][v.1]) * input.d[v.0][v.1] as f64;
        }
        score_delta
    }

    fn action_swap(&mut self, c_a: usize, c_b: usize, input: &Input) -> f64 {
        let mut score_delta = 0.;
        // 取り除く
        for c_i in [c_a, c_b] {
            if self.cycle_usage[c_i] == UNUSED {
                continue;
            }
            score_delta += self.remove_cycle(c_i, input);
        }
        self.cycle_usage.swap(c_a, c_b);

        // 追加する
        for c_i in [c_a, c_b] {
            if self.cycle_usage[c_i] == UNUSED {
                continue;
            }
            score_delta += self.insert_cycle(c_i, input);
        }

        score_delta
    }

    fn action_new_cycle(
        &mut self,
        c_i: usize,
        s: (usize, usize),
        dist: &Vec<Vec<Vec<Vec<i64>>>>,
        input: &Input,
        adj: &Adj,
    ) -> f64 {
        let gain_size: usize = input.n * input.n / 12;
        let mut score_delta = 0.;
        score_delta += self.remove_cycle(c_i, input);

        let mut gain_cand = vec![];
        let start_t = self.cycle_usage[c_i] as f64 / self.cycles.len() as f64;
        let end_t = (self.cycle_usage[c_i] as f64 + 1.) / self.cycles.len() as f64;
        for i in 0..input.n {
            for j in 0..input.n {
                gain_cand.push((
                    calc_delta(FloatIndex((start_t + end_t) / 2.), &self.ps[i][j])
                        * input.d[i][j] as f64,
                    (i, j),
                ));
            }
        }
        gain_cand.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut v = vec![s];
        for i in 0..gain_size {
            v.push(gain_cand[i].1);
        }

        let cycle = create_single_cycle(&v, start_t, end_t, dist, &mut self.ps, input, adj);
        let new_c_i = self.cycles.len();
        self.cycles.push(cycle);
        self.cycle_usage[new_c_i] = c_i;
        score_delta += self.insert_cycle(new_c_i, input);
        score_delta
    }
}

fn optimize_cycles(
    cycle_cnt: usize,
    ideal_cycle_l: usize,
    total_cycle_length: i64,
    cycles: Vec<Vec<(usize, usize)>>,
    input: &Input,
) -> Vec<usize> {
    let mut state = State::new(cycles, total_cycle_length, input);

    let mut iter_count = 0;
    while time::elapsed_seconds() < TIME_LIMIT {
        let (c_a, c_b) = (
            rnd::gen_range(0, state.cycles.len()),
            rnd::gen_range(0, state.cycles.len()),
        );
        if c_a == c_b || (state.cycle_usage[c_a] == UNUSED && state.cycle_usage[c_b] == UNUSED) {
            continue;
        }
        iter_count += 1;

        let prev_score = state.score;

        state.score += state.action_swap(c_a, c_b, input);

        if state.score < prev_score {
            // eprintln!("adopt: {prev_score} -> {score} {c_a} {c_b}");
        } else {
            state.score += state.action_swap(c_a, c_b, input);
        }
    }

    eprintln!("iter_count: {}", iter_count);

    let mut cycle_order = vec![0; cycle_cnt];
    for (i, status) in state.cycle_usage.iter().enumerate() {
        if *status == UNUSED {
            continue;
        }
        cycle_order[*status] = i;
    }
    cycle_order
}
