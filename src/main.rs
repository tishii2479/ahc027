mod def;
mod util;

use crate::def::*;
use crate::util::*;

fn main() {
    time::start_clock();
    let input = Input::read_input();
    eprintln!("{:?}", input.h);

    eprintln!("result: {{\"duration\": {}}}", time::elapsed_seconds());
}
