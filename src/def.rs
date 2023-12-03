use proconio::input;

#[derive(Debug)]
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
            h: [String; n - 1],
            v: [String; n],
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Up,
    Right,
    Down,
    Left,
}

impl Dir {
    pub fn to_char(&self) -> char {
        match self {
            Dir::Up => 'U',
            Dir::Right => 'R',
            Dir::Down => 'D',
            Dir::Left => 'L',
        }
    }

    pub fn rev(&self) -> Dir {
        match self {
            Dir::Up => Dir::Down,
            Dir::Right => Dir::Left,
            Dir::Down => Dir::Up,
            Dir::Left => Dir::Right,
        }
    }

    pub fn add(&self, s: (usize, usize)) -> (usize, usize) {
        match self {
            Dir::Up => (s.0 - 1, s.1),
            Dir::Right => (s.0, s.1 + 1),
            Dir::Down => (s.0 + 1, s.1),
            Dir::Left => (s.0, s.1 - 1),
        }
    }
}
