use polars::prelude::*;
use std::fs::File;
use std::io::{Read, Result};

pub fn read_u32_be(buf: &[u8]) -> u32 {
    u32::from_be_bytes(buf.try_into().unwrap())
}

pub fn load_labels(path: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let magic = read_u32_be(&buf[0..4]);
    assert_eq!(magic, 2049);

    let num = read_u32_be(&buf[4..8]) as usize;
    Ok(buf[8..8 + num].to_vec())
}
pub fn load_images(path: &str) -> Result<Vec<Vec<u8>>> {
    let mut file = File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let magic = read_u32_be(&buf[0..4]);
    assert_eq!(magic, 2051);

    let num = read_u32_be(&buf[4..8]) as usize;
    let rows = read_u32_be(&buf[8..12]) as usize;
    let cols = read_u32_be(&buf[12..16]) as usize;

    let image_size = rows * cols;
    let mut images = Vec::with_capacity(num);

    let mut offset = 16;
    for _ in 0..num {
        images.push(buf[offset..offset + image_size].to_vec());
        offset += image_size;
    }

    Ok(images)
}

pub fn mnist_to_df(images: Vec<Vec<u8>>, labels: Vec<u8>, limit: usize) -> PolarsResult<DataFrame> {
    let n = images.len().min(limit);
    println!("{}", labels.len());
    assert!(labels.len() >= n);

    let mut chunked_builder =
        ListPrimitiveChunkedBuilder::<Float64Type>::new("".into(), 1, 784, DataType::Float64);
    images.into_iter().take(n).for_each(|img| {
        let im: Vec<f64> = img.into_iter().map(|p| p as f64 / 255.0).collect();
        chunked_builder.append_slice(&im);
    });

    let pixels = chunked_builder.finish().into_series();
    let labels = labels.into_iter().take(n).collect::<Vec<u8>>();

    DataFrame::new(vec![
        Column::new("labels".into(), labels),
        Column::new("pixels".into(), pixels),
    ])
}
