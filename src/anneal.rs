const UNUSED: usize = usize::MAX;

struct State {
    score: f64,
    cycle_cnt: usize,
    cycles: Vec<Vec<(usize, usize)>>,
    cycle_usage: Vec<usize>,
    ps: Vec<Vec<BTreeSet<FloatIndex>>>,
}

impl State {
    fn new(cycles: Vec<Vec<(usize, usize)>>, input: &Input) -> State {
        let cycle_cnt = cycles.len();
        let cycle_usage = (0..cycle_cnt).collect();
        let mut ps: Vec<Vec<BTreeSet<FloatIndex>>> = vec![vec![BTreeSet::new(); input.n]; input.n];

        for c_i in 0..cycle_cnt {
            let start_t = c_i as f64 / cycle_cnt as f64;
            let end_t = (c_i as f64 + 1.) / cycle_cnt as f64;
            for (i, v) in cycles[c_i].iter().enumerate() {
                ps[v.0][v.1].insert(to_float_index(start_t, end_t, i, cycles[c_i].len()));
            }
        }

        let mut score = 0.;
        for i in 0..input.n {
            for j in 0..input.n {
                for x in ps[i][j].iter() {
                    score += calc_prev_delta(*x, &ps[i][j]) * input.d[i][j] as f64;
                }
            }
        }

        State {
            score,
            cycle_cnt,
            cycles,
            cycle_usage,
            ps,
        }
    }

    fn remove_cycle(&mut self, c_i: usize, input: &Input) {
        let t = self.cycle_usage[c_i];
        assert_ne!(t, UNUSED);
        let start_t = t as f64 / self.cycle_cnt as f64;
        let end_t = (t as f64 + 1.) / self.cycle_cnt as f64;
        for (i, v) in self.cycles[c_i].iter().enumerate() {
            let index = to_float_index(start_t, end_t, i, self.cycles[c_i].len());
            self.score -= calc_delta(index, &self.ps[v.0][v.1]) * input.d[v.0][v.1] as f64;
            assert!(self.ps[v.0][v.1].remove(&index),);
        }
    }

    fn insert_cycle(&mut self, c_i: usize, input: &Input) {
        let t = self.cycle_usage[c_i];
        assert_ne!(t, UNUSED);
        let start_t = t as f64 / self.cycle_cnt as f64;
        let end_t = (t as f64 + 1.) / self.cycle_cnt as f64;
        for (i, v) in self.cycles[c_i].iter().enumerate() {
            let index = to_float_index(start_t, end_t, i, self.cycles[c_i].len());
            assert!(self.ps[v.0][v.1].insert(index),);
            self.score += calc_delta(index, &self.ps[v.0][v.1]) * input.d[v.0][v.1] as f64;
        }
    }

    fn action_swap(&mut self, c_a: usize, c_b: usize, input: &Input) {
        // 取り除く
        for c_i in [c_a, c_b] {
            if self.cycle_usage[c_i] == UNUSED {
                continue;
            }
            self.remove_cycle(c_i, input);
        }
        self.cycle_usage.swap(c_a, c_b);

        // 追加する
        for c_i in [c_a, c_b] {
            if self.cycle_usage[c_i] == UNUSED {
                continue;
            }
            self.insert_cycle(c_i, input);
        }
    }

    fn action_new_cycle(
        &mut self,
        c_i: usize,
        s: (usize, usize),
        dist: &Vec<Vec<Vec<Vec<i64>>>>,
        input: &Input,
        adj: &Adj,
    ) -> usize {
        let gain_size: usize = input.n * input.n / 12;
        self.remove_cycle(c_i, input);

        let mut gain_cand = vec![];
        let start_t = self.cycle_usage[c_i] as f64 / self.cycle_cnt as f64;
        let end_t = (self.cycle_usage[c_i] as f64 + 1.) / self.cycle_cnt as f64;
        for i in 0..input.n {
            for j in 0..input.n {
                gain_cand.push((
                    -calc_delta(
                        FloatIndex(start_t * ALPHA + end_t * (1. - ALPHA)),
                        &self.ps[i][j],
                    ) * input.d[i][j] as f64,
                    (i, j),
                ));
            }
        }
        gain_cand.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let mut v = vec![s];
        for i in 0..gain_size {
            v.push(gain_cand[i].1);
        }

        let (cycle, score_delta) =
            create_single_cycle(&v, start_t, end_t, dist, &mut self.ps, input, adj);
        let new_c_i = self.cycles.len();
        self.cycles.push(cycle);
        self.cycle_usage.push(self.cycle_usage[c_i]);
        self.cycle_usage[c_i] = UNUSED;
        self.score += score_delta;

        new_c_i
    }
}

fn optimize_cycles(
    cycles: Vec<Vec<(usize, usize)>>,
    s: (usize, usize),
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    adj: &Adj,
    input: &Input,
) -> State {
    let mut state = State::new(cycles, input);

    let mut iter_count = 0;
    while time::elapsed_seconds() < TIME_LIMIT {
        iter_count += 1;
        let prev_score = state.score;

        if rnd::nextf() < 0. {
            let (c_a, c_b) = (
                rnd::gen_range(0, state.cycles.len()),
                rnd::gen_range(0, state.cycles.len()),
            );
            if c_a == c_b || (state.cycle_usage[c_a] == UNUSED && state.cycle_usage[c_b] == UNUSED)
            {
                continue;
            }
            state.action_swap(c_a, c_b, input);

            if state.score < prev_score {
                eprintln!("swap adopt: {prev_score} -> {} {c_a} {c_b}", state.score);
            } else {
                state.action_swap(c_a, c_b, input);
                assert!((state.score - prev_score).abs() < 0.1);
            }
        } else {
            let c_i = rnd::gen_range(0, state.cycles.len());
            if state.cycle_usage[c_i] == UNUSED {
                continue;
            }
            let new_c_i = state.action_new_cycle(c_i, s, dist, input, adj);
            if state.score < prev_score {
                eprintln!("new adopt: {} -> {}", prev_score, state.score);
            } else {
                state.action_swap(c_i, new_c_i, input);
                assert!((state.score - prev_score).abs() < 0.1);
            }
        }
    }

    eprintln!("iter_count: {}", iter_count);

    state
}
