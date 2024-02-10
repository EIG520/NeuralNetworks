use std::path::Path;

pub trait Model: Sized {
    type Shape;

    fn predict(&self, _: Vec<f32>) -> Vec<f32>;
    fn fit(&mut self, _: Vec<Vec<f32>>, _: Vec<Vec<f32>>, _:usize);
    fn new(_:Self::Shape) -> Self;

    fn train_from_csv<P>(&mut self, path: P, input_columns: Vec<usize>, output_columns: Vec<usize>) -> Result<(), csv::Error> 
    where P: AsRef<Path>
    {
        println!("starting training from csv");

        let mut inputs: Vec<Vec<String>> = vec![];
        let mut outputs: Vec<Vec<String>> = vec![];


        let mut reader = csv::Reader::from_path(path)?;

        reader
            .records()
            .try_for_each(|r| -> Option<()> {
                let record: Vec<String> = r.expect("Couldn't open all items in file").into_iter().map(|s| s.to_owned()).collect();

                let mut cur_in: Vec<String> = vec![];
                let mut cur_out: Vec<String> = vec![];

                for &value in &input_columns {
                    cur_in.push((*record.get(value)?).clone());
                }
                for &value in &output_columns {
                    cur_out.push((*record.get(value)?).clone());
                }
                
                inputs.push(cur_in);
                outputs.push(cur_out);

                println!("{:?}", record.get(0));

                Some(())
            });
        
        let remapped_inputs = tokenize(inputs);
        let remapped_outputs = tokenize(outputs);

        self.fit(remapped_inputs, remapped_outputs, 10);

        Ok(())
    }
}

pub fn tokenize(inputs: Vec<Vec<String>>) -> Vec<Vec<f32>> {
    drop(inputs);
    todo!()
}