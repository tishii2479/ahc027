use proconio::input;

pub struct Input {
    pub n: usize,
    pub h: Vec<Vec<bool>>,
    pub v: Vec<Vec<bool>>,
    pub d: Vec<Vec<i64>>,
}

impl Input {
    pub fn read_input() -> Input {
        input! {
            n: usize,
            h: [String; n],
            v: [String; n-1],
            d: [[i64; n]; n],
        }
        let h = h
            .iter()
            .map(|s| {
                s.chars()
                    .map(|c| if c == '1' { true } else { false })
                    .collect()
            })
            .collect();
        let v = v
            .iter()
            .map(|s| {
                s.chars()
                    .map(|c| if c == '1' { true } else { false })
                    .collect()
            })
            .collect();
        Input { n, h, v, d }
    }
}
