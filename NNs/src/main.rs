use nns::{network::*, activations::LeakyReLU};
use std::time::Instant;

fn main() {
    let mut net = TrainableFFNet {
        layers: vec![
            TLayer::<LeakyReLU> {
                nodes: vec![
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                ],
                activation: LeakyReLU {},
            },
            TLayer::<LeakyReLU> {
                nodes: vec![
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                ],
                activation: LeakyReLU {},
            },
            TLayer::<LeakyReLU> {
                nodes: vec![
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                    TNode {
                        weights: vec![1.0,2.0,3.0],
                        value: 0.0,
                        gradients: vec![0.0,0.0,0.0],
                        gradient: 0.0,
                    },
                ],
                activation: LeakyReLU {},
            },
        ]
    };
    

    for i in 0..10000 {
        net.train_one_case(&vec![1.0,2.0,3.0], &vec![0.0,0.0,0.0], 0.0001);
    }
}
