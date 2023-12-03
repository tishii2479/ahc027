mod def;
mod solver;
mod util;

use crate::def::*;
use crate::solver::*;
use crate::util::*;

fn main() {
    time::start_clock();
    let input = Input::read_input();
    let ans = solve(&input);
    println!("{ans}");

    eprintln!(
        "result: {{\"score\": {}, \"duration\": {}}}",
        0,
        time::elapsed_seconds()
    );
}
