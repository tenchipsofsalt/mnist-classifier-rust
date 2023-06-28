extern crate rand;
extern crate tch;

use rand::distributions::{Distribution, Uniform};
use tch::{Tensor, Cuda, vision};
use std::env;

// Hyperparams
const N_CLASSES: usize = 10;
const INPUT_SIZE: usize = 28 * 28;
const STEPS: u32 = 200;
const LR: f32 = 0.1;
const BATCH_SIZE: i64 = 200;


fn main() {

    // // Print one image (the one at index 5) for verification.
    // print_image(&mnist.train_data[5], mnist.train_labels[5]);
    env::set_var("RUST_BACKTRACE", "1");

    let cudnn = Cuda::cudnn_is_available();
    Cuda::set_user_enabled_cudnn(true);
    println!("CUDNN status: {cudnn}. Is your GPU on?");
    let dev = tch::Device::cuda_if_available();

    // mnist
    let m = vision::mnist::load_dir("data").unwrap();
    let train_x = m.train_images.to(dev);
    let train_y = m.train_labels.to(dev);
    let test_x = m.test_images.to(dev);
    let test_y = m.test_labels.to(dev);

    // init weights
    let mut w1 = init_uniform(INPUT_SIZE, N_CLASSES);
    let mut b1 = init_uniform(1, 1);
    let opt = (tch::Kind::Int, dev);

    for epoch in 0..STEPS {
        // make batches
        let batch_permutation = Tensor::randperm(train_x.size()[0], opt);

        // loop over n iterations
        for batch_no in 0..(train_x.size()[0] / BATCH_SIZE) {
            // get batch indices and index into train data
            let batch_idxs = Tensor::arange_start(batch_no * BATCH_SIZE, (batch_no + 1) * BATCH_SIZE, opt);
            let train_idxs = batch_permutation.index_select(0, &batch_idxs);
            let batch_x = train_x.index_select(0, &train_idxs);
            let batch_y = train_y.index_select(0, &train_idxs);

            // forward step
            let out = batch_x.matmul(&w1) + b1;

            // update weights

        }

        // epoch stats
    }
}

fn init_uniform(dim1: usize, dim2: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new_inclusive(0.0, 1.0);

    let mut weights: [f64; dim1 * dim2] = [0.0; dim1 * dim2];

    for weight in weights.iter_mut() {
        *weight = dist.sample(&mut rng);
    }
    let tempsor = Tensor::from_slice(&weights);
    let out_shape: Vec<i64> = vec![dim1 as i64, dim2 as i64];
    let tempsor = tempsor.reshape(&out_shape).set_requires_grad(true);
    println!("Output weight tensor shape: {:?}", out_shape);
    tempsor
}