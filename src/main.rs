use crate::read_byte::{load_images, load_labels, mnist_to_df};
use polars::prelude::*;
mod neural;
mod read_byte;
fn main() -> PolarsResult<()> {
    let images = load_images("train-images-idx3-ubyte")?;
    let labels = load_labels("train-labels-idx1-ubyte")?;
    let df = mnist_to_df(images, labels, 600)?;
    let n_per_layers: Vec<usize> = vec![784, 16, 16, 10];
    let nn = neural::NeuralNetwork::new(n_per_layers);
    nn.train(df)?;

    Ok(())
}
