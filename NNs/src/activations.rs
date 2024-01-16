pub trait Activation: Copy {
    fn apply(&self, _:f32) -> f32;
    fn apply_derivative(&self, _:f32) -> f32;
}

#[derive(Clone, Copy)]
pub struct LeakyReLU {}
impl Activation for LeakyReLU {
    fn apply(&self, n:f32) -> f32 {
        if n > 0.0 {n} else {n / 2.0}
    }
    fn apply_derivative(&self, n:f32) -> f32 {
        if n > 0.0 {1.0} else {0.5}
    }
}
impl Default for LeakyReLU {
    fn default() -> Self {Self {}}
}