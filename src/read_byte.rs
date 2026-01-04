use ndarray::s;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fs::File;
use std::io::{Read, Result};
pub struct DataSet {
    pub labels: Array1<u8>,
    pub images: Array2<u8>,
}

pub struct BatchView<'a> {
    pub images: ArrayView2<'a, u8>, // (batch, 784)
    pub labels: ArrayView1<'a, u8>, // (batch)
}
impl<'a> BatchView<'a> {
    pub fn images_vecf64(&self) -> Array2<f64> {
        let (batch, dim) = self.images.dim();
        let mut buf = Vec::with_capacity(batch * dim);

        for row in self.images.rows() {
            buf.extend(row.iter().map(|p| *p as f64 / 255.0));
        }

        Array2::from_shape_vec((batch, dim), buf).unwrap()
    }
    pub fn label_one_hot(&self) -> Array2<f64> {
        let batch = self.labels.len();
        let mut buf = Vec::with_capacity(batch * 10);

        for &label in self.labels.iter() {
            let mut v = vec![0.0; 10];
            v[label as usize] = 1.0;
            buf.extend(v);
        }

        Array2::from_shape_vec((batch, 10), buf).unwrap()
    }
}
impl DataSet {
    pub fn new(image_path: &str, label_path: &str) -> Result<Self> {
        let images = Self::load_images_ndarray(image_path)?;
        let labels = Self::load_labels(label_path)?;

        assert_eq!(images.nrows(), labels.len());

        Ok(Self { images, labels })
    }
    fn read_u32_be(buf: &[u8]) -> u32 {
        u32::from_be_bytes(buf.try_into().unwrap())
    }

    fn load_labels(path: &str) -> Result<Array1<u8>> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let magic = Self::read_u32_be(&buf[0..4]);
        assert_eq!(magic, 2049);

        let num = Self::read_u32_be(&buf[4..8]) as usize;
        let arr = Array1::from_vec(buf[8..8 + num].to_vec());
        Ok(arr)
    }
    fn load_images_ndarray(path: &str) -> Result<Array2<u8>> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let magic = Self::read_u32_be(&buf[0..4]);
        assert_eq!(magic, 2051);

        let num = Self::read_u32_be(&buf[4..8]) as usize;
        let rows = Self::read_u32_be(&buf[8..12]) as usize;
        let cols = Self::read_u32_be(&buf[12..16]) as usize;

        let image_size = rows * cols;
        let offset = 16;

        let data = &buf[offset..offset + num * image_size];

        Ok(Array2::from_shape_vec((num, image_size), data.to_vec()).unwrap())
    }
    pub fn batch_view_iter(
        &self,
        batch_size: usize,
        limit: usize,
    ) -> impl Iterator<Item = BatchView<'_>> {
        let n = self.labels.len().min(limit);

        (0..n).step_by(batch_size).map(move |start| {
            let end = (start + batch_size).min(n);

            BatchView {
                labels: self.labels.slice(s![start..end]),
                images: self.images.slice(s![start..end, ..]),
            }
        })
    }
}
