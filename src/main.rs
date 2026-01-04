use polars::prelude::*;
extern crate blas_src;
use crate::read_byte::DataSet;
mod neural;
mod read_byte;
fn main() -> PolarsResult<()> {
    let ds = DataSet::new("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
    let ds_test = DataSet::new("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")?;
    let n_per_layers: Vec<usize> = vec![784, 128, 128, 10];
    let mut nn = neural::NeuralNetwork::new(n_per_layers, 0.05);
    let mut trained_time = 1;
    for epoch in 0..10 {
        for (batch_idx, batch) in ds.batch_view_iter(64, 60_000).enumerate() {
            let images = batch.images_vecf64();
            let labels = batch.label_one_hot();
            let (cost, loss) = nn.train(&images, &labels)?;

            // if batch_idx % 100 == 0 {
            println!("epoch:              {}/10", epoch + 1);
            println!("batch:              {}", batch_idx);
            println!("train times:        {}", trained_time);
            println!("C                 = {}", cost);
            println!("loss              = {}", loss);
            println!("Learning Rate     = {:?}", nn.get_learning_rate());
            trained_time += 1;
        }
        if epoch == 1 {
            nn.set_learning_rate(0.005);
        }
    }

    let mut total_correct = 0f64;

    for (batch_idx, batch) in ds_test.batch_view_iter(100, 10_000).enumerate() {
        let images = batch.images_vecf64();
        let labels = batch.label_one_hot();

        let correct = nn.test(&images, &labels); // จำนวนที่ทายถูก
        total_correct += correct;
        let total_samples = images.nrows();

        println!("batch:              {}", batch_idx);
        println!("correct (batch):    {}", correct * total_samples as f64);
    }

    let accuracy = total_correct * 100.0;
    println!("Accuracy = {:.2}%", accuracy);

    Ok(())
}
