extern crate rand;
extern crate tch;

use rand::distributions::{Distribution, Uniform};
use tch::{Tensor, Cuda, vision};
use std::env;

// Hyperparams
const CLASSES: usize = 10;
const INPUT_SIZE: usize = 28 * 28;
const STEPS: u32 = 200;
const LR: f32 = 0.1;



const WEIGHTS_UNROLLED_SIZE: usize = (INPUT_SIZE + 1) * CLASSES;

fn main() {

    // // Print one image (the one at index 5) for verification.
    // print_image(&mnist.train_data[5], mnist.train_labels[5]);
    env::set_var("RUST_BACKTRACE", "1");

    let cuda = Cuda::cudnn_is_available();
    println!("CUDA status: {cuda}. Is your GPU on?");

    // mnist
    let m = vision::mnist::load_dir("data").unwrap();
    let x = m.train_images;
    let y = m.train_labels;

    // init weights
    let mut w = init_weights_uniform();
    // w.print()

    'train: for i in 0..STEPS {
        // compute images x weights in batches

        // update weights

        // aggregate statistics
    }
}

fn init_weights_uniform() -> Tensor {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new_inclusive(0.0, 1.0);

    let mut weights: [f64; WEIGHTS_UNROLLED_SIZE] = [0.0; WEIGHTS_UNROLLED_SIZE];

    for weight in weights.iter_mut() {
        *weight = dist.sample(&mut rng);
    }
    let tempsor = Tensor::from_slice(&weights);
    let out_shape: Vec<i64> = vec![(INPUT_SIZE + 1) as i64, CLASSES as i64];
    let tempsor = tempsor.reshape(&out_shape).set_requires_grad(true);
    println!("Output weight tensor shape: {:?}", out_shape);
    tempsor
}