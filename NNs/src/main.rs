use nns::{network::*, model::*};

fn main() {
    let mut net = TrainableFFNet::new(vec![16, 1]);

    let mut inputs: Vec<Vec<f32>> = vec![];
    let outputs: Vec<Vec<f32>> = vec![
        0, 1, 1, 1,
        1, 1, 0, 0,
        1, 1, 1, 1,
        0, 1, 0, 1
    ].iter().map(|&i| vec![i as f32]).collect();

    for i in 0..16 {
        let mut input = vec![0.0;16];
        input[i] = 1.0;

        inputs.push(input);
    }
    
    for _ in 0..1000 {
        net.train_on(&inputs, &outputs);
                

        for i in 0..16 {
            let mut input = vec![0.0;16];
            input[i] = 1.0;

            if i % 4 == 0 {println!();}
            print!("{:.3} ", net.gen_out(input)[0]);
        }
        println!();
    }
}


fn working_example() {
    let mut net = TrainableFFNet::new(vec![2, 1]);
    for _ in 0..1000000 {
        net.train_one_case_adam(&vec![0.0,1.0], &vec![1.0], 0.0001, 0.9, 0.999, 0.00000001);
        
        net.train_one_case_adam(&vec![1.0,0.0], &vec![2.0], 0.0001, 0.9, 0.999, 0.00000001);

        net.print_value();
        
        println!("(0,1) => {:?}", net.gen_out(vec![0.0, 1.0]));
        println!("(1,0) => {:?}", net.gen_out(vec![1.0, 0.0]));
        
        println!();
    }

    println!("(0,1) => {:?}", net.gen_out(vec![0.0,1.0]));
    println!("(1,0) => {:?}", net.gen_out(vec![1.0,0.0]));
}