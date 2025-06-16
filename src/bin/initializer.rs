use ndarray::{Array, Array2, ArrayD};
use ndarray_rand::{
    RandomExt,
    rand_distr::{Normal, Uniform},
};

/// 神经网络参数初始化方法
pub enum Initializer {
    Zeros,         // 全零初始化
    Ones,          // 全1初始化
    RandomUniform, // 均匀分布随机初始化
    RandomNormal,  // 正态分布随机初始化
    Xavier,        // Xavier/Glorot 初始化
    XavierUniform, // Xavier/Glorot 均匀分布初始化
    He,            // He/Kaiming 初始化
    HeUniform,     // He/Kaiming 均匀分布初始化
}

impl Initializer {
    /// 初始化二维权重矩阵
    pub fn initialize_weights(&self, input_size: usize, output_size: usize) -> Array2<f32> {
        match self {
            Initializer::Zeros => Array::zeros((input_size, output_size)),
            Initializer::Ones => Array::ones((input_size, output_size)),
            Initializer::RandomUniform => {
                Array::random((input_size, output_size), Uniform::new(-1.0, 1.0))
            }
            Initializer::RandomNormal => {
                Array::random((input_size, output_size), Normal::new(0.0, 1.0).unwrap())
            }
            Initializer::Xavier => {
                let scale = (2.0 / (input_size + output_size) as f32).sqrt();
                Array::random((input_size, output_size), Normal::new(0.0, scale).unwrap())
            }
            Initializer::XavierUniform => {
                let limit = (6.0f32 / (input_size + output_size) as f32).sqrt();
                Array::random((input_size, output_size), Uniform::new(-limit, limit))
            }
            Initializer::He => {
                let scale = (2.0 / input_size as f32).sqrt();
                Array::random((input_size, output_size), Normal::new(0.0, scale).unwrap())
            }
            Initializer::HeUniform => {
                let limit = (6.0f32 / input_size as f32).sqrt();
                Array::random((input_size, output_size), Uniform::new(-limit, limit))
            }
        }
    }

    /// 初始化偏置向量
    pub fn initialize_biases(&self, size: usize) -> ArrayD<f32> {
        match self {
            Initializer::Zeros => Array::zeros(size).into_dyn(),
            Initializer::Ones => Array::ones(size).into_dyn(),
            Initializer::RandomUniform => Array::random(size, Uniform::new(-1.0, 1.0)).into_dyn(),
            Initializer::RandomNormal => {
                Array::random(size, Normal::new(0.0, 1.0).unwrap()).into_dyn()
            }
            Initializer::Xavier
            | Initializer::He
            | Initializer::XavierUniform
            | Initializer::HeUniform => {
                // 通常偏置初始化为0，即使使用Xavier或He初始化
                Array::zeros(size).into_dyn()
            }
        }
    }
}

fn main() {
    // 初始化一个 3x4 的权重矩阵
    let input_size = 3;
    let output_size = 4;

    println!("Zero initialization:");
    let weights_zero = Initializer::Zeros.initialize_weights(input_size, output_size);
    println!("{:?}", weights_zero);

    println!("\nOnes initialization:");
    let weights_ones = Initializer::Ones.initialize_weights(input_size, output_size);
    println!("{:?}", weights_ones);

    println!("\nRandom Uniform initialization:");
    let weights_uniform = Initializer::RandomUniform.initialize_weights(input_size, output_size);
    println!("{:?}", weights_uniform);

    println!("\nXavier initialization:");
    let weights_xavier = Initializer::Xavier.initialize_weights(input_size, output_size);
    println!("{:?}", weights_xavier);

    println!("\nXavier Uniform initialization:");
    let weights_xavier = Initializer::XavierUniform.initialize_weights(input_size, output_size);
    println!("{:?}", weights_xavier);

    println!("\nHe initialization:");
    let weights_he = Initializer::He.initialize_weights(input_size, output_size);
    println!("{:?}", weights_he);

    println!("\nHe Uniform initialization:");
    let weights_he = Initializer::HeUniform.initialize_weights(input_size, output_size);
    println!("{:?}", weights_he);

    // 初始化偏置
    println!("\nBias initialization (zeros):");
    let biases = Initializer::Zeros.initialize_biases(output_size);
    println!("{:?}", biases);
}
