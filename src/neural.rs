use std::vec;

use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use polars::prelude::*;
#[derive(Debug)]
pub struct NeuralNetwork {
    bias_matrix: Vec<Array1<f64>>,
    weight_matrix: Vec<Array2<f64>>,
    n_layers: usize,
    learning_rate: f64,
    trained_time: usize,
}
impl NeuralNetwork {
    pub fn new(node_per_layer: Vec<usize>, learning_rate: f64) -> Self {
        let bias_matrix: Vec<Array1<f64>> = node_per_layer
            .iter()
            .skip(1)
            .map(|&n| Array1::<f64>::zeros(n))
            .collect();

        let weight_matrix: Vec<Array2<f64>> = node_per_layer
            .windows(2)
            .map(|w| {
                let (i, j) = (w[0], w[1]);
                let scale = (2.0 / i as f64).sqrt();
                Array::random((i, j), Normal::new(0.0, scale).unwrap())
            })
            .collect();
        let n_layers = node_per_layer.len();
        Self {
            bias_matrix,
            weight_matrix,
            n_layers,
            learning_rate,
            trained_time: 1,
        }
    }

    pub fn test(&mut self, input: DataFrame) -> PolarsResult<()> {
        let input_set = input.clone().lazy().select([col("pixels")]).collect()?;
        let mut chunked_builder =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("".into(), 1, 10, DataType::Float64);
        input_set
            .get_columns()
            .first()
            .unwrap()
            .phys_iter()
            .for_each(|a0| {
                let mut a = self.anyvalue_to_array1(&a0);
                let mut v_z_rn_l: Vec<Array1<f64>> = Vec::new();
                let mut v_node_rn_l: Vec<Array1<f64>> = Vec::new();
                for layer in 0..self.n_layers - 1 {
                    v_node_rn_l.push(a.clone());
                    a = self.feedforward(a, layer);
                    v_z_rn_l.push(a.clone());
                }
                chunked_builder.append_slice(a.as_slice().unwrap());
            });

        let series = chunked_builder.finish().into_series();
        let cols: Vec<Column> = vec![
            Column::new("results".into(), series),
            input.column("labels")?.to_owned(),
        ];
        let results = DataFrame::new(cols)?;
        let results = self.diff(results)?;
        let results = self.create_labels(results)?;

        let results = self.is_collect(results)?;
        let results = results
            .lazy()
            .with_columns([(col("label accuracy").mean() * lit(100.0)).alias("accuracy_percent")])
            .collect()?;
        println!("{}", results.head(Some(1)));
        Ok(())
    }
    pub fn train(&mut self, input: &DataFrame) -> PolarsResult<()> {
        let input_set = input.clone().lazy().select([col("pixels")]).collect()?;
        let mut chunked_builder =
            ListPrimitiveChunkedBuilder::<Float64Type>::new("".into(), 1, 10, DataType::Float64);
        let mut v_z_any_l: Vec<Vec<Array1<f64>>> = Vec::new();
        let mut v_node_any_l: Vec<Vec<Array1<f64>>> = Vec::new();
        input_set
            .get_columns()
            .first()
            .unwrap()
            .phys_iter()
            .for_each(|a0| {
                let mut a = self.anyvalue_to_array1(&a0);
                let mut v_z_rn_l: Vec<Array1<f64>> = Vec::new();
                let mut v_node_rn_l: Vec<Array1<f64>> = Vec::new();
                for layer in 0..self.n_layers - 1 {
                    v_node_rn_l.push(a.clone());
                    a = self.feedforward(a, layer);
                    v_z_rn_l.push(a.clone());
                }
                v_node_any_l.push(v_node_rn_l);
                v_z_any_l.push(v_z_rn_l);
                chunked_builder.append_slice(a.as_slice().unwrap());
            });

        let series = chunked_builder.finish().into_series();
        let cols: Vec<Column> = vec![
            Column::new("results".into(), series),
            input.column("labels")?.to_owned(),
        ];

        let result = DataFrame::new(cols)?;

        let result = self.cost(result)?;

        let result = self.c_i(result)?;

        let result = self.c_0(result)?;
        let result = result
            .lazy()
            .clone()
            .with_columns([col("C_0").sum().alias("C")])
            .collect()?;

        let cd_i = result
            .clone()
            .lazy()
            .clone()
            .with_columns([col("Cd_i")])
            .collect()?["Cd_i"]
            .clone();
        let v_cd_i: Vec<Array1<f64>> = cd_i
            .phys_iter()
            .map(|c| self.anyvalue_to_array1(&c))
            .collect();
        let size_of_dataset = result.height();

        let mut sum_dc_d_weight: Vec<Array2<f64>> = (0..self.n_layers - 1)
            .map(|l| Array2::zeros(self.weight_matrix[l].raw_dim()))
            .collect();
        let mut sum_dc_d_bias: Vec<Array1<f64>> = (0..self.n_layers - 1)
            .map(|l| Array1::zeros(self.bias_matrix[l].len()))
            .collect();
        for i in 0..size_of_dataset {
            let cd = v_cd_i[i].clone(); // 2(a^L - y)
            let z = v_z_any_l[i][self.n_layers - 2].clone();
            let a = v_node_any_l[i][self.n_layers - 2].clone();
            let mut delta = self.delta(cd.clone(), z.clone());
            let dc_dw = self.pd_c_by_weight_arr(delta.clone(), a.clone());
            sum_dc_d_weight[self.n_layers - 2] += &dc_dw;
            sum_dc_d_bias[self.n_layers - 2] += &delta;
            for layer in (0..=self.n_layers - 3).rev() {
                delta = self.delta_w(
                    self.weight_matrix[layer + 1].clone(),
                    delta.clone(),
                    v_z_any_l[i][layer].clone(),
                );

                let dc_dw = v_node_any_l[i][layer]
                    .clone()
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&delta.view().insert_axis(Axis(0)));
                sum_dc_d_weight[layer] += &dc_dw;
                sum_dc_d_bias[layer] += &delta;
            }
        }

        let scale = self.learning_rate / size_of_dataset as f64;

        for (w, dw) in self.weight_matrix.iter_mut().zip(sum_dc_d_weight.iter()) {
            *w -= &(dw * scale);
        }

        for (b, db) in self.bias_matrix.iter_mut().zip(sum_dc_d_bias.iter()) {
            *b -= &(db * scale);
        }
        let c = result
            .lazy()
            .select([col("C"), col("C_0").mean().alias("loss")])
            .collect()?;

        let c = c.get_row(0)?.clone();
        let c_val = c.0[0].clone();
        let loss_val = c.0[1].clone();

        println!("C                 = {}", c_val.to_string());
        println!("loss              = {}", loss_val.to_string());
        println!("Learning Rate     = {:?}", self.learning_rate);
        println!("Trainned Times    = {:?}", self.trained_time);
        self.trained_time += 1;
        Ok(())
    }
    fn cost(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("results"), col("labels")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_results = ca.field_by_name("results")?;
                        let series_labels = ca.field_by_name("labels")?;
                        let chunked_results = series_results.list()?;
                        let chunked_labels = series_labels.u8()?;

                        let create_expect_vec = |n: usize| {
                            let mut arr = vec![0.; 10];
                            arr[n] = 1.;
                            Series::new("".into(), arr)
                        };
                        let gradiat_oprate = |result, label| -> PolarsResult<Series> {
                            let diff: Series = (result - create_expect_vec(label))?;
                            let sqre = (diff.clone() * diff.clone())?;
                            Ok(sqre)
                        };
                        let out: ListChunked = chunked_results
                            .into_iter()
                            .zip(chunked_labels)
                            .map(|(result, label)| match (result, label) {
                                (Some(a), Some(b)) => {
                                    let v = gradiat_oprate(a, b as usize).ok()?;
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| {
                        Ok(Field::new(
                            f.name().clone(),
                            DataType::List(Box::new(
                                Field::new("".into(), DataType::Float64).dtype,
                            )),
                        ))
                    },
                )
                .alias("C_i")])
            .collect()?;
        Ok(res)
    }

    fn c_i(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("results"), col("labels")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_results = ca.field_by_name("results")?;
                        let series_correctness = ca.field_by_name("labels")?;
                        let chunked_results = series_results.list()?;
                        let chunked_labels = series_correctness.u8()?;

                        let create_expect_vec = |n: usize| {
                            let mut arr = vec![0.; 10];
                            arr[n] = 1.;
                            Series::new("".into(), arr)
                        };
                        let gradiat_oprate = |result, label| -> PolarsResult<Series> {
                            let diff: Series = (result - create_expect_vec(label))?;
                            let sqre = diff.clone() * 2;
                            Ok(sqre)
                        };
                        let out: ListChunked = chunked_results
                            .into_iter()
                            .zip(chunked_labels)
                            .map(|(result, label)| match (result, label) {
                                (Some(a), Some(b)) => {
                                    let v = gradiat_oprate(a, b as usize).ok()?;
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| {
                        Ok(Field::new(
                            f.name().clone(),
                            DataType::List(Box::new(
                                Field::new("".into(), DataType::Float64).dtype,
                            )),
                        ))
                    },
                )
                .alias("Cd_i")])
            .collect()?;
        Ok(res)
    }

    fn is_collect(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("results"), col("label vec")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_results = ca.field_by_name("results")?;
                        let series_correctness = ca.field_by_name("label vec")?;
                        let chunked_results = series_results.list()?;
                        let chunked_labels = series_correctness.list()?;

                        let gradiat_oprate = |result, label| -> bool { result == label };
                        let out: BooleanChunked = chunked_results
                            .into_iter()
                            .zip(chunked_labels)
                            .map(|(result, label)| match (result, label) {
                                (Some(a), Some(b)) => {
                                    let v = gradiat_oprate(a, b);
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| Ok(Field::new(f.name().clone(), DataType::Boolean)),
                )
                .alias("label accuracy")])
            .collect()?;
        Ok(res)
    }
    fn diff(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("results"), col("labels")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_results = ca.field_by_name("results")?;
                        let series_correctness = ca.field_by_name("labels")?;
                        let chunked_results = series_results.list()?;
                        let chunked_labels = series_correctness.u8()?;

                        let create_expect_vec = |n: usize| {
                            let mut arr = vec![0.; 10];
                            arr[n] = 1.;
                            Series::new("".into(), arr)
                        };
                        let gradiat_oprate = |result, label| -> PolarsResult<Series> {
                            let diff: Series = (result - create_expect_vec(label))?;
                            Ok(diff)
                        };
                        let out: ListChunked = chunked_results
                            .into_iter()
                            .zip(chunked_labels)
                            .map(|(result, label)| match (result, label) {
                                (Some(a), Some(b)) => {
                                    let v = gradiat_oprate(a, b as usize).ok()?;
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| {
                        Ok(Field::new(
                            f.name().clone(),
                            DataType::List(Box::new(
                                Field::new("".into(), DataType::Float64).dtype,
                            )),
                        ))
                    },
                )
                .alias("diff")])
            .collect()?;
        Ok(res)
    }
    fn c_0(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("results"), col("labels")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_results = ca.field_by_name("results")?;
                        let series_correctness = ca.field_by_name("labels")?;
                        let chunked_results = series_results.list()?;
                        let chunked_labels = series_correctness.u8()?;

                        let create_expect_vec = |n: usize| {
                            let mut arr = vec![0.; 10];
                            arr[n] = 1.;
                            Series::new("".into(), arr)
                        };
                        let gradiat_oprate = |result, label| -> PolarsResult<f64> {
                            let diff: Series = (result - create_expect_vec(label))?;
                            let sqre = (diff.clone() * diff.clone())?;
                            sqre.sum()
                        };
                        let out: Float64Chunked = chunked_results
                            .into_iter()
                            .zip(chunked_labels)
                            .map(|(result, label)| match (result, label) {
                                (Some(a), Some(b)) => {
                                    let v = gradiat_oprate(a, b as usize).ok()?;
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| Ok(Field::new(f.name().clone(), DataType::Float64)),
                )
                .alias("C_0")])
            .collect()?;
        Ok(res)
    }

    fn create_labels(&self, results: DataFrame) -> PolarsResult<DataFrame> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("labels")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_correctness = ca.field_by_name("labels")?;
                        let chunked_labels = series_correctness.u8()?;

                        let create_expect_vec = |n: usize| {
                            let mut arr = vec![0.; 10];
                            arr[n] = 1.;
                            Series::new("".into(), arr)
                        };
                        let out: ListChunked = chunked_labels
                            .into_iter()
                            .map(|label| match label {
                                Some(a) => {
                                    let v = create_expect_vec(a as usize);
                                    Some(v)
                                }
                                _ => None,
                            })
                            .collect();

                        Ok(out.into_column())
                    },
                    |_, f| {
                        Ok(Field::new(
                            f.name().clone(),
                            DataType::List(Box::new(
                                Field::new("".into(), DataType::Float64).dtype,
                            )),
                        ))
                    },
                )
                .alias("label vec")])
            .collect()?;
        Ok(res)
    }
    fn delta_w(&self, w: Array2<f64>, delta: Array1<f64>, z: Array1<f64>) -> Array1<f64> {
        w.dot(&delta) * z
    }
    fn delta(&self, c: Array1<f64>, z: Array1<f64>) -> Array1<f64> {
        c * z.mapv(Self::relu_prime)
    }
    fn pd_c_by_weight_arr(&self, delta: Array1<f64>, ak: Array1<f64>) -> Array2<f64> {
        ak.view()
            .insert_axis(Axis(1))
            .dot(&delta.view().insert_axis(Axis(0)))
    }

    fn feedforward(&self, a: Array1<f64>, layer: usize) -> Array1<f64> {
        (a.dot(&self.weight_matrix[layer]) + &self.bias_matrix[layer]).mapv(Self::relu)
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

    pub fn set_learning_rate(&mut self, rl: f64) {
        self.learning_rate = rl;
    }
    pub fn decesses(&mut self, rl: f64) {
        self.learning_rate -= rl;
    }
    fn anyvalue_to_array1(&self, s: &AnyValue<'_>) -> Array1<f64> {
        match s {
            AnyValue::List(s) => {
                let v: Vec<f64> = s.f64().unwrap().into_no_null_iter().collect();
                Array1::from(v)
            }
            _ => panic!("expected List[f64]"),
        }
    }
    fn detect_grad_norm(norm: f64, layer: usize, name: &str) {
        if norm < 1e-8 {
            println!(
                "☠️  DEAD GRADIENT | layer {} {} | norm = {:.3e}",
                layer, name, norm
            );
        } else if norm > 1e3 {
            println!(
                "🔥 EXPLODING GRADIENT | layer {} {} | norm = {:.3e}",
                layer, name, norm
            );
        }
    }

    fn grad_norm_1d(g: &Array1<f64>) -> f64 {
        g.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    fn grad_norm_2d(g: &Array2<f64>) -> f64 {
        g.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}
