use std::vec;

use ndarray::{Array, Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use polars::frame::row::Row;
use polars::prelude::*;
#[derive(Debug)]
pub struct NeuralNetwork {
    bias_matrix: Vec<Array1<f64>>,
    weight_matrix: Vec<Array2<f64>>,
    n_layers: usize,
    learning_rate: f64,
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
        }
    }
    pub fn train(&mut self, input: DataFrame) -> PolarsResult<()> {
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
        let mut is_first = true;
        for i in 0..size_of_dataset {
            let mut cd = v_cd_i[i].clone(); // 2(a^L - y)
            for layer in (0..self.n_layers - 1).rev() {
                let z = v_z_any_l[i][layer].clone();
                let a = v_node_any_l[i][layer].clone();
                let dc_dw = self.pd_c_by_weight_arr(cd.clone(), z.clone(), a.clone());
                let dc_db = self.pd_c_by_bais_arr(cd.clone(), z.clone());
                let dc_da =
                    self.pd_c_by_ak_arr(cd.clone(), z.clone(), self.weight_matrix[layer].clone());

                sum_dc_d_weight[layer] += &dc_da;
                sum_dc_d_bias[layer] += &dc_db;
                cd = 2f64 * (a - dc_dw);
            }
        }
        // println!("{:#?}", sum_dc_d_weight);
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
        println!("{}", c.head(Some(1)));
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
    fn agg_cost_per_instance(&self, results: DataFrame) -> PolarsResult<Array1<f64>> {
        let res = results
            .clone()
            .lazy()
            .with_columns([as_struct(vec![col("sigma_i")])
                .map(
                    |s| {
                        let ca = s.struct_()?;
                        let series_correctness = ca.field_by_name("sigma_i")?;
                        let chunked_correctness = series_correctness.list()?;

                        let arr = vec![0f64; 10];
                        let mut sum_c = Series::new("".into(), arr);
                        let out: ListChunked = chunked_correctness
                            .into_iter()
                            .map(|correctness| match correctness {
                                Some(a) => {
                                    sum_c = (sum_c.clone() + a).ok()?;
                                    Some(sum_c.clone())
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
                .alias("sum C_0")])
            .collect()?;
        let agg = res
            .clone()
            .lazy()
            .select([(col("sum C_0") / col("sigma_i").count()).alias("avg C_0")])
            .last()
            .collect()?;
        let row = agg.get_row(0)?;
        let arr: Array1<f64> = match &row.0[0] {
            AnyValue::List(series) => Array1::from_iter(series.f64()?.into_no_null_iter()),
            _ => unreachable!(),
        };

        Ok(arr)
    }

    fn pd_c_by_weight_arr(&self, c: Array1<f64>, z: Array1<f64>, ak: Array1<f64>) -> Array1<f64> {
        c.dot(&z.mapv(Self::relu_prime)) * (ak)
    }

    fn pd_c_by_ak_arr(&self, c: Array1<f64>, z: Array1<f64>, w: Array2<f64>) -> Array2<f64> {
        c * z.mapv(Self::relu_prime) * w
    }

    fn pd_c_by_bais_arr(&self, c: Array1<f64>, z: Array1<f64>) -> Array1<f64> {
        c * (z.mapv(Self::relu_prime))
    }
    fn pd_c_by_bais(&self, c: f64, z: f64) -> f64 {
        c * Self::relu_prime(z)
    }

    fn pd_c_by_ak(&self, c: f64, z: f64, w: f64) -> f64 {
        c * Self::relu_prime(z) * w
    }
    fn pd_c_by_weight(&self, c: f64, z: f64, ak: f64) -> f64 {
        c * Self::relu_prime(z) * ak
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

    fn anyvalue_to_array1(&self, s: &AnyValue<'_>) -> Array1<f64> {
        match s {
            AnyValue::List(s) => {
                let v: Vec<f64> = s.f64().unwrap().into_no_null_iter().collect();
                return Array1::from(v);
            }
            _ => panic!("expected List[f64]"),
        }
    }

    fn anyvalue_to_f64(&self, s: &AnyValue<'_>) -> f64 {
        match s {
            AnyValue::Float64(s) => *s,
            _ => panic!("expected List[f64]"),
        }
    }
    fn row_to_array1(&self, s: &Row<'_>) -> Array1<f64> {
        match s {
            Row(s) => {
                let v: Vec<f64> = s.iter().map(|x| self.anyvalue_to_f64(x)).collect();
                return Array1::from(v);
            }
            _ => panic!("expected List[f64]"),
        }
    }
    pub fn print(&self) {
        println!("bais :{:#?}", self.bias_matrix[0]);
        println!("weight :{:#?}", self.weight_matrix[0]);
    }

    pub fn print_shape(&self) {
        for i in self.weight_matrix.clone() {
            println!("weight :{:#?}", i.shape());
        }
    }
}
