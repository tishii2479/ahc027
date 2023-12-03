use std::collections::BinaryHeap;
use std::collections::{BTreeSet, VecDeque};

use crate::def::*;
use crate::util::*;

type Adj = Vec<Vec<Vec<(Dir, (usize, usize))>>>;

const INF: i64 = 1e12 as i64;
const EDGE_WEIGHT: i64 = 1e6 as i64; // NOTE: USED_CODEより小さくあるべき
const USED_CODE: i64 = -1e9 as i64;
const GETA: i64 = 1e5 as i64;

pub fn solve(input: &Input) -> String {
    const L: usize = 1e4 as usize;
    let r = calc_r(input);
    let adj = create_adj(input);
    let mut dist = vec![vec![vec![]; input.n]; input.n];
    let mut s = (0, 0);
    let mut count_set: BTreeSet<(i64, (usize, usize))> = BTreeSet::new();
    let mut remain_count = vec![vec![0; input.n]; input.n];

    for i in 0..input.n {
        for j in 0..input.n {
            dist[i][j] = calc_dist((i, j), input, &adj);

            remain_count[i][j] = (L as f64 * r[i][j]).round() as i64;
            count_set.insert((GETA, (i, j)));

            if r[i][j] > r[s.0][s.1] {
                (s.0, s.1) = (i, j);
            }
        }
    }

    let required_count = remain_count.clone();

    show(&remain_count);

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
            &mut count_set,
            &mut remain_count,
            &required_count,
            input,
            &adj,
        );
        cycles.push(cycle);
    }

    show(&remain_count);
    rnd::shuffle(&mut cycles);

    eprintln!("s:           {:?}", s);
    eprintln!("cycle_l:     {cycle_l}");
    eprintln!("cycle_cnt:   {cycle_cnt}");

    cycles_to_answer(&cycles)
}

fn create_cycle(
    s: (usize, usize),
    cycle_l: i64,
    dist: &Vec<Vec<Vec<Vec<i64>>>>,
    count_set: &mut BTreeSet<(i64, (usize, usize))>,
    remain_count: &mut Vec<Vec<i64>>,
    required_count: &Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    let mut v = (s.0, s.1);
    let mut path = vec![];
    let mut rev_counts = vec![];

    while cycle_l - path.len() as i64 > dist[s.0][s.1][v.0][v.1] {
        let mut last = vec![];
        let mut iter = count_set.iter();
        let cand_size = if path.len() == 0 {
            1
        } else {
            input.n * input.n / 16
        };
        for _ in 0..cand_size {
            let (_, v) = iter.next_back().unwrap();
            last.push(*v);
        }

        assert!(last.len() > 0);
        let target_v = last
            .iter()
            .filter(|&x| x != &v)
            .min_by(|&x, &y| dist[v.0][v.1][x.0][x.1].cmp(&dist[v.0][v.1][y.0][y.1]))
            .unwrap();

        let add_path = shortest_path(&v, target_v, &remain_count, input, &adj);
        for v in add_path.iter() {
            if remain_count[v.0][v.1] == USED_CODE {
                continue;
            }
            rev_counts.push((*v, remain_count[v.0][v.1] - 1));
            assert!(
                count_set.remove(&(GETA * remain_count[v.0][v.1] / required_count[v.0][v.1], *v))
            );
            remain_count[v.0][v.1] = USED_CODE;
            count_set.insert((USED_CODE, *v));
        }
        path.extend(add_path);
        v = *target_v;
    }

    // vからsに戻る
    let return_path = shortest_path(&v, &s, &remain_count, input, &adj);
    for v in return_path.iter() {
        if remain_count[v.0][v.1] == USED_CODE {
            continue;
        }
        rev_counts.push((*v, remain_count[v.0][v.1] - 1));
        assert!(count_set.remove(&(GETA * remain_count[v.0][v.1] / required_count[v.0][v.1], *v)));
        remain_count[v.0][v.1] -= 1;
        count_set.insert((GETA * remain_count[v.0][v.1] / required_count[v.0][v.1], *v));
    }
    path.extend(return_path);

    for (v, rev_t) in rev_counts {
        count_set.remove(&(USED_CODE, v));
        remain_count[v.0][v.1] = rev_t;
        count_set.insert((GETA * remain_count[v.0][v.1] / required_count[v.0][v.1], v));
    }

    path
}

fn show(remain_count: &Vec<Vec<i64>>) {
    eprintln!("-----");
    let mut sum = 0;
    let mut max = 0;
    for i in 0..remain_count.len() {
        for j in 0..remain_count[i].len() {
            eprint!("{:5}", remain_count[i][j]);
            if remain_count[i][j] > 0 {
                max = max.max(remain_count[i][j]);
                sum += remain_count[i][j];
            }
        }
        eprintln!();
    }
    eprintln!("max: {max}");
    eprintln!("ave: {:.5}", sum as f64 / remain_count.len().pow(2) as f64);
}

fn shortest_path(
    s: &(usize, usize),
    t: &(usize, usize),
    remain_count: &Vec<Vec<i64>>,
    input: &Input,
    adj: &Adj,
) -> Vec<(usize, usize)> {
    use std::cmp::Reverse;
    let mut dist = vec![vec![1e17 as i64; input.n]; input.n];
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
            let cost = if remain_count[u.0][u.1] <= 0 {
                INF
            } else {
                EDGE_WEIGHT - remain_count[u.0][u.1]
            };
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
            let cost = if remain_count[cur.0][cur.1] <= 0 {
                INF
            } else {
                EDGE_WEIGHT - remain_count[cur.0][cur.1]
            };
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
