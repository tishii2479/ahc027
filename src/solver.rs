use std::collections::{BTreeSet, VecDeque};

use crate::def::*;
use crate::util::*;

type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

pub fn solve(input: &Input) -> String {
    const L: usize = 1e4 as usize;
    const USED_CODE: i64 = -1e9 as i64;
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);
    let mut counts: BTreeSet<(i64, (usize, usize))> = BTreeSet::new();
    let mut t = vec![vec![0; input.n]; input.n];

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);

            t[i][j] = (L as f64 * r[i][j]).round() as i64;
            counts.insert((t[i][j], (i, j)));

            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    for i in 0..input.n {
        for j in 0..input.n {
            eprint!("{:5}", t[i][j]);
        }
        eprintln!();
    }

    let s = s;
    let cycle_l = (1. / r[s.0][s.1]).round() as usize;
    let cycle_cnt = L / cycle_l;
    let mut cycles = vec![];

    // サイクルの作成
    for _ in 0..cycle_cnt {
        let mut v = (s.0, s.1);
        let mut path = vec![];
        let mut rev_counts = vec![];
        while cycle_l - path.len() > dist[s.0][s.1][v.0][v.1] as usize {
            let mut last = vec![];
            let mut iter = counts.iter();
            for _ in 0..input.n * input.n / 16 {
                let (_, v) = iter.next_back().unwrap();
                last.push(*v);
            }

            assert!(last.len() > 0);
            let target_v = last
                .iter()
                .filter(|&x| x != &v)
                .min_by(|&x, &y| dist[v.0][v.1][x.0][x.1].cmp(&dist[v.0][v.1][y.0][y.1]))
                .unwrap();

            let next_dir = get_move_cand(v, *target_v, &dist, &adj)
                .iter()
                .max_by(|&x, &y| {
                    let x = x.add(v);
                    let y = y.add(v);
                    t[x.0][x.1].cmp(&t[y.0][y.1])
                })
                .unwrap()
                .to_owned();
            v = next_dir.add(v);
            path.push(v);

            if t[v.0][v.1] == USED_CODE {
                continue;
            }
            counts.remove(&(t[v.0][v.1], v));
            rev_counts.push((v, t[v.0][v.1] - 1));
            t[v.0][v.1] = USED_CODE;
            counts.insert((USED_CODE, v));
        }

        // vからsに戻る
        while v != s {
            let next_dir = get_move_cand(v, s, &dist, &adj)
                .iter()
                .max_by(|&x, &y| {
                    let x = x.add(v);
                    let y = y.add(v);
                    t[x.0][x.1].cmp(&t[y.0][y.1])
                })
                .unwrap()
                .to_owned();
            v = next_dir.add(v);
            path.push(v);

            if t[v.0][v.1] == USED_CODE {
                continue;
            }
            counts.remove(&(t[v.0][v.1], v));
            rev_counts.push((v, t[v.0][v.1] - 1));
            t[v.0][v.1] = USED_CODE;
            counts.insert((USED_CODE, v));
        }

        for (v, rev_t) in rev_counts {
            counts.remove(&(USED_CODE, v));
            t[v.0][v.1] = rev_t;
            counts.insert((rev_t, v));
        }
        cycles.push(path);
    }

    rnd::shuffle(&mut cycles);

    eprintln!("-----");
    for i in 0..input.n {
        for j in 0..input.n {
            eprint!("{:5}", t[i][j]);
        }
        eprintln!();
    }

    eprintln!("s:           {:?}", s);
    eprintln!("cycle_l:     {cycle_l}");
    eprintln!("cycle_cnt:   {cycle_cnt}");

    cycles_to_answer(&cycles)
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
