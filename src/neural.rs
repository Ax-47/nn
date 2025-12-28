use ndarray::{Array, Array1, Array2, ArrayView1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use polars::prelude::*;
#[derive(Debug)]
pub struct NeuralNetwork {
    bias_matrix: Vec<Array1<f32>>,
    weight_matrix: Vec<Array2<f32>>,
    n_layers: usize,
}
impl NeuralNetwork {
    pub fn new(node_per_layer: Vec<usize>) -> Self {
        let bias_matrix: Vec<Array1<f32>> = node_per_layer
            .iter()
            .skip(1)
            .map(|&n| Array1::<f32>::zeros(n))
            .collect();

        let weight_matrix: Vec<Array2<f32>> = node_per_layer
            .windows(2)
            .map(|w| {
                let (i, j) = (w[0], w[1]);
                let scale = (2.0 / i as f32).sqrt();
                Array::random((i, j), Normal::new(0.0, scale).unwrap())
            })
            .collect();
        let n_layers = node_per_layer.len();
        Self {
            bias_matrix,
            weight_matrix,
            n_layers,
        }
    }
    pub fn train(&self, input: DataFrame) -> PolarsResult<()> {
        let input_set = input
            .lazy()
            .select([col("^pixel_.*$")])
            .collect()?
            .to_ndarray::<Float32Type>(IndexOrder::C)
            .unwrap();
        for a0 in input_set.outer_iter() {
            let b = Array::from_iter(a0);
            let mut a: Array1<f32> = b.mapv(|x| *x);
            for layer in 0..self.n_layers - 1 {
                a = self.feedforward(a, layer);
            }
            println!("{a}");
        }
        Ok(())
    }

    fn feedforward(&self, a: Array1<f32>, layer: usize) -> Array1<f32> {
        (a.dot(&self.weight_matrix[layer]) + &self.bias_matrix[layer]).mapv(Self::relu)
    }
    fn relu(x: f32) -> f32 {
        if x < 0. {
            return 0.;
        }
        x
    }
    pub fn print(&self) {
        println!("bais :{:#?}", self.bias_matrix);
        println!("weight :{:#?}", self.weight_matrix);
    }
}
