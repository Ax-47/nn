use crate::read_byte::{load_images, load_labels, mnist_batch_iter, mnist_to_df};
use polars::prelude::*;
mod neural;
mod read_byte;
fn main() -> PolarsResult<()> {
    let images = load_images("train-images-idx3-ubyte")?;
    let labels = load_labels("train-labels-idx1-ubyte")?;
    let n_per_layers: Vec<usize> = vec![784, 128, 16, 16, 16, 10];
    // let n_per_layers: Vec<usize> = vec![784, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 10];
    let df = mnist_to_df(images.clone(), labels.clone(), 60)?;
    let mut nn = neural::NeuralNetwork::new(n_per_layers, 0.05);
    for epoch in 0..10 {
        for batch in mnist_batch_iter(&images, &labels, 64, 60_000) {
            let df = batch?;
            nn.train(&df)?;
        }

        nn.test(df.clone())?;

        if epoch == 3 {
            nn.set_learning_rate(0.005);
            break;
        }
    }
    Ok(())
}
