use crate::activations::*;

const DEFAULT_LAYER_SIZE: usize = 1024;

#[derive(Clone)]
pub struct Node {
    weights: Vec<f32>,
}

impl Node {
    pub fn gen_val_from_inputs(&self, inputs: &Vec<f32>) -> f32 {
        inputs.iter().enumerate().map(|(i,f)| f * self.weights[i]).sum()
    }
}

impl Default for Node {
    fn default() -> Self {
        Self {
            weights: vec![-1.0; DEFAULT_LAYER_SIZE]
        }
    }
}

#[derive(Clone)]
pub struct Layer<T: Activation> {
    nodes: Vec<Node>,
    activation: T,
}

impl<T: Activation> Layer<T> {
    fn step(&self, input: &Vec<f32>) -> Vec<f32> {
        self.nodes.iter().map(
            |node| self.activation.apply(node.gen_val_from_inputs(input))
        ).collect()
    }
}

impl<T: Activation + Default> Default for Layer<T> {
    fn default() -> Self {
        Self {
            nodes:vec![Node::default(); DEFAULT_LAYER_SIZE],
            activation:T::default()
        }
    }
}

pub struct FFNet {
    layers: Vec<Layer<LeakyReLU>>,
}

impl FFNet {
    pub fn gen_out(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers.iter().fold(
            inputs,
            |input, layer| layer.step(&input)
        )
    }
}

impl Default for FFNet {
    fn default() -> Self {
        Self {
            layers: vec![ Layer::default(); 100]
        }
    }
}

// Trainable net
// Probably slightly less performant because it is trainable

#[derive(Clone)]
pub struct TNode {
    pub weights: Vec<f32>,
    pub value: f32,
    pub gradients: Vec<f32>,
    pub gradient: f32,
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
            self.gradients[i] = self.weights[i] * act.apply_derivative(self.value) * self.gradient;
            prev.update_node_gradient(i, self.gradients[i]);
        }
    }

    pub fn move_by_gradient(&mut self, lr: f32) {
        for i in 0..self.weights.len() {
            self.weights[i] -= self.gradients[i] * lr;
        }
    }

    pub fn print_value(&self) {print!("{} ", self.value);}
}

impl Default for TNode {
    fn default() -> Self {
        Self {
            weights: vec![-1.0; DEFAULT_LAYER_SIZE],
            value: 0.0,
            gradients: vec![0.0; DEFAULT_LAYER_SIZE],
            gradient: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct TLayer<T: Activation> {
    pub nodes: Vec<TNode>,
    pub activation: T,
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

    pub fn print_value(&self) {
        for node in self.nodes.clone() {
            node.print_value();
        }
        println!();
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
    pub layers: Vec<TLayer<LeakyReLU>>,
}

impl TrainableFFNet {
    pub fn gen_out(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers.iter().fold(
            inputs,
            |input, layer| layer.step(&input)
        )
    }

    pub fn forward_pass(&mut self, inputs: &Vec<f32>) {
        let mut cur: Vec<f32> = inputs.clone();
        for layer in &mut self.layers {
            layer.forward_pass(&cur);
            cur = layer.step(&cur);
        }
    }

    pub fn backward_pass(&mut self, expected_outputs: &Vec<f32>) {
        let last = self.layers.len()-1;
        let final_layer = &mut self.layers[last];

        final_layer.gen_gradients_from_expected(expected_outputs);

        let mut layers = self.layers.iter_mut().rev();

        //layers.next();
        
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

    pub fn train_one_case(&mut self, inputs: &Vec<f32>, expected_outputs: &Vec<f32>, lr: f32) {
        self.forward_pass(inputs);
        self.print_value();println!();
        self.backward_pass(expected_outputs);
        self.move_by_gradient(lr);
    }

    pub fn print_value(&self) {
        for layer in self.layers.clone() {
            layer.print_value();
        }
    }
}

impl Default for TrainableFFNet {
    fn default() -> Self {
        Self {
            layers: vec![ TLayer::default(); 100]
        }
    }
}