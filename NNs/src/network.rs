use crate::activations::*;
use crate::model::*;
use rand::prelude::*;

const DEFAULT_LAYER_SIZE: usize = 3;

// Trainable net

#[derive(Clone)]
pub struct TNode {
    weights: Vec<f32>,
    value: f32,
    gradients: Vec<f32>,
    gradient: f32,

    first_moment_vectors: Vec<f32>,
    second_moment_vectors: Vec<f32>,
    iteration_count: i32,
}

impl TNode {
    pub fn gen_val_from_inputs(&self, inputs: &Vec<f32>) -> f32 {
        inputs.iter().enumerate().map(|(i,f)| f * self.weights[i]).sum()
    }
    pub fn forward_pass<T: Activation>(&mut self, inputs: &Vec<f32>, act: T) {
        self.value = act.apply(inputs.iter().enumerate().map(|(i,f)| f * self.weights[i]).sum());
    }
    pub fn backward_pass<T: Activation, T1: Activation>(&mut self, act: T, prev: &mut TLayer<T1>) {
        for i in 0..self.weights.len() {
            self.gradients[i] = prev.nodes[i].value * act.apply_derivative(self.value) * self.gradient;
            prev.update_node_gradient(i, self.gradients[i]);
        }
    }

    pub fn move_by_gradient(&mut self, lr: f32) {
        for i in 0..self.weights.len() {
            self.weights[i] -= self.gradients[i] * lr;
        }
    }

    pub fn move_adam(&mut self, beta_1: f32, beta_2: f32, alpha: f32, epsilon: f32) {
        //println!("{:?} ", self.weights);

        for i in 0..self.weights.len() {
            self.first_moment_vectors[i] = beta_1 * self.first_moment_vectors[i] + (1.0 - beta_1) * self.gradients[i];
            self.second_moment_vectors[i] = beta_2 * self.second_moment_vectors[i] + (1.0 - beta_2) * self.gradients[i] * self.gradients[i];

            let bias_corrected_first = self.first_moment_vectors[i] / (1.0 - beta_1.powi(self.iteration_count));
            let bias_corrected_second = self.second_moment_vectors[i] / (1.0 - beta_2.powi(self.iteration_count));
            
            //print!("{} ", bias_corrected_second);

            self.weights[i] -= alpha * bias_corrected_first / (bias_corrected_second.sqrt() + epsilon)
        }
    }
    
    pub fn reset_adam_info(&mut self) {
        self.second_moment_vectors = vec![0.0; self.second_moment_vectors.len()];
        self.first_moment_vectors = vec![0.0; self.first_moment_vectors.len()];
        self.iteration_count = 0;
    }

    pub fn print_value(&self) {
        print!("{}: {:?} ", self.value, self.weights.iter().map(|i| format!("{:.3}", i)).collect::<Vec<String>>());
    }
}

impl Default for TNode {
    fn default() -> Self {
        TNode::new(DEFAULT_LAYER_SIZE)
    }
}

impl TNode {
    fn new(weights: usize) -> Self {
        Self {
            weights: vec![0.0; weights].iter()
                .map(|_| rand::thread_rng().gen::<f32>())
                .collect::<Vec<f32>>(),
            value: 0.0,
            gradients: vec![0.0; weights],
            gradient: 0.0,

            first_moment_vectors: vec![0.0; weights],
            second_moment_vectors: vec![0.0; weights],
            iteration_count: 0,
        }
    }
}

#[derive(Clone)]
pub struct TLayer<T: Activation> {
    nodes: Vec<TNode>,
    activation: T,
}

impl<T: Activation> TLayer<T> {
    pub fn step(&self, input: &Vec<f32>) -> Vec<f32> {
        self.nodes.iter().map(
            |node| self.activation.apply(node.gen_val_from_inputs(input))
        ).collect()
    }

    fn forward_pass(&mut self, input: &Vec<f32>) {
        for node in &mut self.nodes {
            node.forward_pass(input, self.activation);
        }
    }

    pub fn reset_adam_info(&mut self) {
        for node in self.nodes.iter_mut() {
            node.reset_adam_info();
        }
    }

    fn backward_pass<T2: Activation>(&mut self, prev: &mut TLayer<T2>) {
        for node in &mut self.nodes {
            node.backward_pass(self.activation, prev);
        }
    }

    fn update_node_gradient(&mut self, node: usize, val: f32) {
        self.nodes[node].gradient += val;
    }

    fn reset_node_gradients(&mut self) {
        for node in &mut self.nodes {node.gradient = 0.0;}
    }

    fn gen_gradients_from_expected(&mut self, expected: &Vec<f32>) {
        for i in 0..self.nodes.len() {
            let node = &mut self.nodes[i];
            //let squared_error = (node.value - expected[i]) * (node.value - expected[i]);

            node.gradient = 2.0 * (node.value - expected[i]);
        }
    }

    fn move_by_gradient(&mut self, lr: f32) {
        for node in &mut self.nodes {
            node.move_by_gradient(lr);
        }
    }

    fn move_adam(&mut self, alpha: f32, beta_1: f32, beta_2: f32, epsilon: f32) {
        for node in &mut self.nodes {
            node.iteration_count += 1;
            node.move_adam(beta_1, beta_2, alpha, epsilon);
        }
    }

    pub fn print_value(&self) {
        for node in self.nodes.clone() {
            node.print_value();
        }
        println!();
    }

    pub fn set_values(&mut self, to: &Vec<f32>) {
        for i in 0..to.len() {
            self.nodes[i].value = to[i];
        }
    }

    pub fn values(&mut self) -> Vec<f32> {
        let mut values = vec![];

        for node in &mut self.nodes {
            values.push(node.value);
        }

        values
    }
}

impl<T: Activation + Default> TLayer<T> {
    pub fn new(prev_size: usize, size: usize, activation: T) -> TLayer<T> {
        Self {
            nodes:vec![TNode::new(0); size].iter()
                .map(|_| TNode::new(prev_size))
                .collect::<Vec<TNode>>(),
            activation
        }
    }
}

impl<T: Activation + Default> Default for TLayer<T> {
    fn default() -> Self {
        Self {
            nodes:vec![TNode::default(); DEFAULT_LAYER_SIZE],
            activation:T::default()
        }
    }
}

pub struct TrainableFFNet {
    layers: Vec<TLayer<LeakyReLU>>,
}

impl TrainableFFNet {
    pub fn gen_out(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers.iter().skip(1).fold(
            inputs,
            |input, layer| layer.step(&input)
        )
    }

    pub fn forward_pass(&mut self, inputs: &Vec<f32>) {
        self.layers[0].set_values(inputs);

        for i in 1..self.layers.len() {
            let pvals = self.layers[i-1].values();
            let layer = &mut self.layers[i];

            layer.forward_pass(&pvals);
        }
    }

    pub fn backward_pass(&mut self, expected_outputs: &Vec<f32>) {
        let last = self.layers.len()-1;
        let mut first_layer = vec![TLayer::new(0, self.layers[0].nodes[0].weights.len(), LeakyReLU {})];
        let final_layer = &mut self.layers[last];

        final_layer.gen_gradients_from_expected(expected_outputs);

        let mut layers = self.layers.iter_mut().rev().chain(first_layer.iter_mut());
        
        let mut l = layers.next();
        let mut prev = layers.next();

        while let Some(pre) = prev.as_mut() {

            if let Some(layer) = l {
                pre.reset_node_gradients();
                layer.backward_pass(pre);
                
                l = prev;
                prev = layers.next();
            }
        }
    }

    pub fn move_by_gradient(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.move_by_gradient(lr);
        }
    }

    pub fn move_adam(&mut self, alpha: f32, beta_1:f32, beta_2: f32, epsilon: f32) {
        for layer in &mut self.layers {
            layer.move_adam(alpha, beta_1, beta_2, epsilon);
        }
    }

    pub fn train_one_case_stochastic_gradient_descent(&mut self, inputs: &Vec<f32>, expected_outputs: &Vec<f32>, lr: f32) {
        self.forward_pass(inputs);
        self.backward_pass(expected_outputs);
        self.move_by_gradient(lr);
    }

    pub fn train_one_case_adam(&mut self, inputs: &Vec<f32>, expected_outputs: &Vec<f32>, alpha: f32, beta_1:f32, beta_2: f32, epsilon: f32) {
        //self.reset_adam_info();
        //for _ in 0..iters {
            self.forward_pass(inputs);
            self.backward_pass(expected_outputs);
            self.move_adam(alpha, beta_1, beta_2, epsilon);
        //}
    }

    pub fn reset_adam_info(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.reset_adam_info();
        }
    }

    pub fn train_on(&mut self, inputs: &Vec<Vec<f32>>, expected_outputs: &Vec<Vec<f32>>) {
        for (input,expected) in inputs.into_iter().zip(expected_outputs) {
            self.train_one_case_adam(input, expected,0.0001, 0.9, 0.999, 0.00000001)
        }
    }

    pub fn print_value(&self) {
        for layer in self.layers.clone() {
            layer.print_value();
        }
    }

}

impl TrainableFFNet {

}

impl Model for TrainableFFNet {
    type Shape = Vec<usize>;

    fn predict(&self, input: Vec<f32>) -> Vec<f32> {
        self.gen_out(input)
    }
    fn fit(&mut self, inputs: Vec<Vec<f32>>, expected: Vec<Vec<f32>>, epochs: usize) {
        for _ in 0..epochs {
            self.train_on(&inputs, &expected);
        }
    }

    fn new(shape: Vec<usize>) -> Self {
        let slf = Self {
            layers: shape.iter()
                .zip((0..=0)
                    .chain(shape.iter().map(|&i| i))
                )
                .map(|(&c, p)| TLayer::new(p, c, LeakyReLU {}))
                .collect()
        };

        slf
    }
}

impl Default for TrainableFFNet {
    fn default() -> Self {
        Self {
            layers: vec![ TLayer::default(); 100 ]
        }
    }
}