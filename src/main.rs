mod def;
mod solver;
mod solver_util;
mod util;

use crate::def::*;
use crate::solver::*;
use crate::util::*;

fn path_to_answer(path: &Vec<(usize, usize)>) -> String {
    let mut ans = vec![];
    for i in 0..path.len() {
        ans.push(Dir::from(path[i], path[(i + 1) % path.len()]));
    }
    ans.iter().map(|d| d.to_char()).collect()
}

fn main() {
    time::start_clock();
    let input = Input::read_input();
    let path = solve(&input);
    let ans = path_to_answer(&path);
    println!("{ans}");

    let score = calc_score(&input, &path);

    eprintln!(
        "result: {{\"score\": {}, \"duration\": {}}}",
        score,
        time::elapsed_seconds()
    );
}
