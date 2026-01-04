use ndarray::{Array, Array1, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use polars::prelude::*;
#[derive(Debug)]
pub struct NeuralNetwork {
    bias_matrix: Vec<Array1<f64>>,
    weight_matrix: Vec<Array2<f64>>,
    n_layers: usize,
    node_per_layers: Vec<usize>,
    learning_rate: f64,
}
impl NeuralNetwork {
    pub fn new(node_per_layers: Vec<usize>, learning_rate: f64) -> Self {
        let bias_matrix: Vec<Array1<f64>> = node_per_layers
            .iter()
            .skip(1)
            .map(|&n| Array1::<f64>::zeros(n))
            .collect();

        let weight_matrix: Vec<Array2<f64>> = node_per_layers
            .windows(2)
            .map(|w| {
                let (i, j) = (w[0], w[1]);
                let scale = (2.0 / i as f64).sqrt();
                Array::random((i, j), Normal::new(0.0, scale).unwrap())
            })
            .collect();
        let n_layers = node_per_layers.len();
        Self {
            bias_matrix,
            weight_matrix,
            n_layers,
            learning_rate,
            node_per_layers,
        }
    }

    pub fn test(&self, images: &Array2<f64>, labels: &Array2<f64>) -> f64 {
        let batch = images.nrows();
        let outputs = self.predict(images); // (batch, n_classes)
        let mut correct = 0;
        for i in 0..batch {
            let pred = self.argmax(&outputs.row(i).to_owned());
            let truth = self.argmax(&labels.row(i).to_owned());
            if pred == truth {
                correct += 1;
            }
        }
        correct as f64 / images.nrows() as f64
    }
    pub fn train(
        &mut self,
        images: &Array2<f64>,
        labels: &Array2<f64>,
    ) -> PolarsResult<(f64, f64)> {
        let mut a = images.clone(); // (layers, batch)
        let batch = images.nrows();
        let max_n = self.max_neurons();

        let mut activations = Array3::<f64>::zeros((self.n_layers, batch, max_n)); //(layers, batch, neurons)
        let mut zs = Array3::<f64>::zeros((self.n_layers, batch, max_n));

        for layer in 0..self.n_layers - 1 {
            let n_a = a.ncols();
            activations.slice_mut(s![layer, .., 0..n_a]).assign(&a);
            let z = self.z_l_b(&a, layer);
            let n_z = z.ncols();
            zs.slice_mut(s![layer, .., 0..n_z]).assign(&z);
            a = self.a_l_b(&z);
        }

        let diff = &a - labels;
        let sq = &diff * &diff;
        let loss = sq.sum();
        let cost = loss / images.nrows() as f64;
        let mut delta_l: Array2<f64> = 2.0 * (&a - labels);

        let scale = self.learning_rate / images.nrows() as f64;

        let n_prev = self.node_per_layers[self.n_layers - 2];
        let a_prev = activations
            .slice(s![self.n_layers - 2, .., 0..n_prev])
            .to_owned();
        let dc_dw = self.pd_c_by_weight_arr_b(&delta_l, &a_prev);
        self.weight_matrix[self.n_layers - 2] -= &(dc_dw * scale);
        self.bias_matrix[self.n_layers - 2] -= &(delta_l.sum_axis(Axis(0)) * scale);
        for layer in (1..self.n_layers - 2).rev() {
            let n_l = self.node_per_layers[layer + 1];
            let z_l = zs.slice(s![layer + 1, .., 0..n_l]).to_owned();
            let n_prev = self.node_per_layers[layer];
            let a_prev = activations.slice(s![layer, .., 0..n_prev]).to_owned();
            delta_l = self.delta_b(&self.weight_matrix[layer + 1], &delta_l, &z_l);
            let dc_dw = self.pd_c_by_weight_arr_b(&delta_l, &a_prev);
            self.weight_matrix[layer] -= &(dc_dw * scale);
            self.bias_matrix[layer] -= &(delta_l.sum_axis(Axis(0)) * scale);
        }
        Ok((cost, loss))
    }
    /// W(j,k) Delta(batch)
    fn delta_b(&self, w: &Array2<f64>, delta: &Array2<f64>, z: &Array2<f64>) -> Array2<f64> {
        delta.dot(&w.t()) * z.mapv(Self::relu_prime)
    }

    fn pd_c_by_weight_arr_b(&self, delta: &Array2<f64>, ak: &Array2<f64>) -> Array2<f64> {
        ak.t().dot(delta)
    }

    /// Z^L=a^L-1_k * w^L + b^L
    fn z_l_b(&self, a: &Array2<f64>, layer: usize) -> Array2<f64> {
        a.dot(&self.weight_matrix[layer]) + &self.bias_matrix[layer]
    }

    /// Z^L=a^L-1_k * w^L + b^L
    fn a_l_b(&self, z: &Array2<f64>) -> Array2<f64> {
        z.mapv(Self::relu)
    }
    fn relu(x: f64) -> f64 {
        if x < 0. {
            return 0.;
        }
        x
    }

    fn relu_prime(x: f64) -> f64 {
        if x < 0. {
            return 0.;
        }
        1f64
    }
    #[inline]
    pub fn max_neurons(&self) -> usize {
        *self
            .node_per_layers
            .iter()
            .max()
            .expect("n_per_layers must not be empty")
    }
    fn argmax(&self, row: &Array1<f64>) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }
    pub fn predict(&self, images: &Array2<f64>) -> Array2<f64> {
        let mut a = images.clone();

        for layer in 0..self.n_layers - 1 {
            let z = self.z_l_b(&a, layer);
            a = self.a_l_b(&z);
        }

        a // (batch, n_classes)
    }
    pub fn set_learning_rate(&mut self, rl: f64) {
        self.learning_rate = rl;
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}
