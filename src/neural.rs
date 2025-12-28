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
}
impl NeuralNetwork {
    pub fn new(node_per_layer: Vec<usize>) -> Self {
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
        }
    }
    pub fn train(&self, input: DataFrame) -> PolarsResult<()> {
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

                for layer in 0..self.n_layers - 1 {
                    a = self.feedforward(a, layer);
                }

                chunked_builder.append_slice(a.as_slice().unwrap());
            });

        let series = chunked_builder.finish().into_series();
        let cols: Vec<Column> = vec![
            Column::new("results".into(), series),
            input.column("labels")?.to_owned(),
        ];

        let result = DataFrame::new(cols)?;

        println!("{}", result.clone().head(Some(5)));
        let result = self.cost(result)?;
        println!("{}", result.clone().head(Some(5)));

        let result = self.c_i(result)?;
        println!("{}", result.clone().head(Some(5)));
        let result = self.agg_cost_per_instance(result)?;

        println!("cost: {}", result.clone());
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
                .alias("sigma_i")])
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
                            Ok(diff * 2)
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
        println!("bais :{:#?}", self.bias_matrix);
        println!("weight :{:#?}", self.weight_matrix);
    }
}
