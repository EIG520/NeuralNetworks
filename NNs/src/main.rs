use nns::{network::*, activations::LeakyReLU};

fn main() {
    let mut net = TrainableFFNet {
        layers: vec![
            TLayer::new(1, 128, LeakyReLU {}),
            TLayer::new(128, 1, LeakyReLU {}),
        ]
    };

    let inputs: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32]).collect();
    let outputs: Vec<Vec<f32>> = inputs.iter().map(|i| vec![i[0] * 2.0]).collect();

    for _ in 0..1000 {
        net.train_on(&inputs, &outputs);
    }

    println!("{:?}", net.gen_out(vec![3462.0]));
}
