pub trait Activation: Copy {
    fn apply(&self, _:f32) -> f32;
    fn apply_derivative(&self, _:f32) -> f32;
}

#[derive(Clone, Copy, Default)]
pub struct LeakyReLU {}
impl Activation for LeakyReLU {
    fn apply(&self, n:f32) -> f32 {
        if n > 0.0 {n} else {n / 2.0}
    }
    fn apply_derivative(&self, n:f32) -> f32 {
        if n >= 0.0 {1.0} else {0.5}
    }
}

#[derive(Clone, Copy, Default)]
pub struct ClippedReLU {}
impl Activation for ClippedReLU {
    fn apply(&self, n:f32) -> f32 {
        if n >= 0.0 {if n <= 1.0 {n} else {1.0}}
        else {0.0}
    }
    fn apply_derivative(&self, n:f32) -> f32 {
        if (0.0..=1.0).contains(&n) {1.0}
        else {0.0}
    }
}