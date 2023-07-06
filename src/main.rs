extern crate rand;
extern crate tch;

use std::env;
use tch::{nn, nn::OptimizerConfig, vision, Cuda, Tensor};

// Hyperparams
const N_CLASSES: i64 = 10;
const INPUT_SIZE: i64 = 28 * 28;
const STEPS: u32 = 1000;
const LR: f64 = 0.001;
const BATCH_SIZE: i64 = 60000;

fn main() {
    // // Print one image (the one at index 5) for verification.
    // print_image(&mnist.train_data[5], mnist.train_labels[5]);
    env::set_var("RUST_BACKTRACE", "1");

    let cudnn = Cuda::cudnn_is_available();
    Cuda::set_user_enabled_cudnn(true);
    println!("CUDNN status: {cudnn}. Is your GPU on?");
    let dev = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(dev);
    let opt = (tch::Kind::Int, dev);

    // mnist
    let m = vision::mnist::load_dir("data").unwrap();
    let train_x = m.train_images.to(dev);
    let train_y = m.train_labels.to(dev).one_hot(N_CLASSES);
    let test_x = m.test_images.to(dev);
    let test_y = m.test_labels.to(dev).one_hot(N_CLASSES);

    // init weights
    // let w1 = init_uniform(INPUT_SIZE, N_CLASSES).set_requires_grad(true);
    // let b1 = init_uniform(1, 1).set_requires_grad(true);
    // let w1 = w1.to(dev);
    // let b1 = b1.to(dev);
    let w1 = vs.root().kaiming_normal("w1", &[INPUT_SIZE, N_CLASSES]);
    let b1 = vs.root().kaiming_normal("b1", &[1, 1]);
    let mut optim = nn::Adam::default().build(&vs, LR).unwrap();

    for epoch in 0..STEPS {
        // make batches
        let batch_permutation = Tensor::randperm(train_x.size()[0], opt);

        let mut total_loss: f64 = 0.0;
        // loop over n iterations
        for batch_no in 0..(train_x.size()[0] / BATCH_SIZE) {
            // get batch indices and index into train data
            let batch_idxs =
                Tensor::arange_start(batch_no * BATCH_SIZE, (batch_no + 1) * BATCH_SIZE, opt);
            let train_idxs = batch_permutation.index_select(0, &batch_idxs);
            let batch_x = train_x.index_select(0, &train_idxs);
            let batch_y = train_y.index_select(0, &train_idxs);
            // forward step
            let out = batch_x.matmul(&w1) + &b1;
            // loss
            let loss = (out.softmax(-1, tch::Kind::Float) - batch_y)
                .pow_(2 as f64)
                .mean(tch::Kind::Float);
            // update weights
            optim.backward_step(&loss);
            total_loss += &loss.detach().double_value(&[]);
        }
        let test_loss = ((&test_x.matmul(&w1) + &b1).softmax(-1, tch::Kind::Float) - &test_y)
            .pow_(2 as f64)
            .mean(tch::Kind::Float)
            .detach()
            .double_value(&[]);
        println!(
            "Epoch {}, train loss: {}, test loss: {}",
            epoch,
            total_loss / (train_x.size()[0] / BATCH_SIZE) as f64,
            test_loss
        )

        // epoch stats
    }
}
