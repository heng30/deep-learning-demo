use ndarray::{Array, Array1, Array2, ArrayD, Ix2, s};
use ndarray_rand::{RandomExt, rand_distr::Normal};

// 激活函数及其导数
pub enum Activation {
    Sigmoid,
    ReLU,
    Tanh,
    Linear,
}

impl Activation {
    fn apply(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::ReLU => x.mapv(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Linear => x.clone(),
        }
    }

    fn derivative(&self, x: &ArrayD<f32>) -> ArrayD<f32> {
        match self {
            Activation::Sigmoid => {
                let sigmoid = self.apply(x);
                &sigmoid * &(ArrayD::<f32>::ones(sigmoid.shape()) - &sigmoid)
            }
            Activation::ReLU => x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
            Activation::Tanh => x.mapv(|v| 1.0 - v.tanh().powi(2)),
            Activation::Linear => Array::ones(x.shape()),
        }
    }
}

// 神经网络层
pub struct DenseLayer {
    pub weights: Array2<f32>,   // 梯度
    pub biases: Array1<f32>,    // 偏置
    pub activation: Activation, // 激活函数

    pub input: Option<ArrayD<f32>>,  // 保存前向传播输入用于反向传播
    pub output: Option<ArrayD<f32>>, // 保存前向传播输出
    pub linear_output: Option<ArrayD<f32>>, // 保存前向传播wx+b的值,用于反向传播
    pub weights_grad: Option<Array2<f32>>, // 移动平均累计历史梯度，用于优化梯度
    pub squared_grad: Option<Array2<f32>>, // RMSProp使用
    pub adaptive_learning_rate: Option<ArrayD<f32>>, // 自适应学习率
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, activation: Activation) -> Self {
        // Xavier/Glorot 初始化权重
        let scale = (2.0 / (input_size + output_size) as f32).sqrt();
        let weights = Array::random((input_size, output_size), Normal::new(0.0, scale).unwrap());

        // 初始化为0的偏置
        let biases = Array::zeros(output_size);

        DenseLayer {
            weights,
            biases,
            activation,
            input: None,
            output: None,
            linear_output: None,
            weights_grad: None,
            squared_grad: None,
            adaptive_learning_rate: None,
        }
    }

    pub fn forward(&mut self, input: &ArrayD<f32>) -> ArrayD<f32> {
        // 将输入保存用于反向传播
        self.input = Some(input.clone());

        // 计算线性部分: Wx + b
        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();
        let linear_output = input_2d.dot(&self.weights) + &self.biases;

        self.linear_output = Some(linear_output.clone().into_dyn());

        // 应用激活函数
        let output = self.activation.apply(&linear_output.into_dyn());
        self.output = Some(output.clone());

        output
    }

    pub fn backward(
        &mut self,
        grad_output: &ArrayD<f32>,
        momentum: f32,
        learning_rate: f32,
        _epochs: usize,
    ) -> ArrayD<f32> {
        // 获取前向传播保存的值
        let input = self.input.as_ref().unwrap();
        let linear_output = self.linear_output.as_ref().unwrap();

        // 计算激活函数的导数
        let activation_derivative = self.activation.derivative(linear_output);

        // 计算当前层的误差, 按照矩阵对应位置一一相乘，不是矩阵乘法
        let delta = grad_output * &activation_derivative;

        // 将输入和delta转换为2D数组以便矩阵运算
        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();
        let delta_2d = delta.clone().into_dimensionality::<Ix2>().unwrap();

        // 计算权重梯度。`z = wx + b`，对`w`求导就是`x`
        // 这里要转置是因为，每一行对应的是同一组数据与不同神经元计算得到的。而delta_2d每一列中的元素对应一组的梯度。出来的结果是，不同组数据对相同权重值的梯度和
        let weights_grad = input_2d.t().dot(&delta_2d);

        // 计算偏置梯度 (对batch取平均)，沿着列求均值
        let biases_grad = delta_2d.mean_axis(ndarray::Axis(0)).unwrap();

        // 计算传递到前一层的梯度。`z = wx + b`，对`x`求导就是`w`
        // 每一组输入数据占据一行输入梯度
        let grad_input = delta_2d.dot(&self.weights.t()).into_dyn();

        // 优化梯度和学习率
        match &self.weights_grad.clone() {
            Some(wg) => {
                // 优化梯度: 移动加权平均动量法
                let weights_grad_momentum = momentum * wg + (1. - momentum) * &weights_grad;
                self.weights_grad = Some(weights_grad_momentum.clone());

                // FIXME: 算法会造成学习率过大
                // RMSProp优化学习率
                // let squared_grad = match &self.squared_grad {
                //     Some(sg) => momentum * sg + (1.0 - momentum) * weights_grad.mapv(|g| g.powi(2)),
                //     None => (1.0 - momentum) * weights_grad.mapv(|g| g.powi(2)),
                // };
                // self.squared_grad = Some(squared_grad.clone());
                //
                // // println!("{:?}", self.squared_grad);
                //
                // // 参数偏差修正
                // let weights_grad_momentum =
                //     weights_grad_momentum / (1.0 - momentum.powf(epochs as f32));
                //
                // let squared_grad = squared_grad / (1.0 - momentum.powf(epochs as f32));
                //
                // let adaptive_learning_rate = match &self.adaptive_learning_rate {
                //     Some(lr) => {
                //         let lr = lr
                //             - (learning_rate / (squared_grad.mapv(|sg| sg.sqrt()) + 1e-6))
                //                 * &weights_grad_momentum;
                //
                //         lr.into_dyn()
                //     }
                //     None => {
                //         let lr = -learning_rate / (squared_grad.mapv(|sg| sg.sqrt()) + 1e-6)
                //             * &weights_grad_momentum;
                //         lr.into_dyn()
                //     }
                // };
                //
                // self.adaptive_learning_rate = Some(adaptive_learning_rate);
                // println!("{:?}", self.adaptive_learning_rate);
            }
            None => {
                self.weights_grad = Some(weights_grad);
            }
        };

        // 更新权重和偏置
        let adaptive_learning_rate = match &self.adaptive_learning_rate {
            Some(lr) => lr,
            None => &(Array::ones(self.weights.shape()) * learning_rate).into_dyn(),
        };

        self.weights -= &(self.weights_grad.as_ref().unwrap() * adaptive_learning_rate);
        self.biases -= &(biases_grad * adaptive_learning_rate.mean_axis(ndarray::Axis(0)).unwrap());

        grad_input
    }
}

// 神经网络模型
pub struct NeuralNetwork {
    layers: Vec<DenseLayer>,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        NeuralNetwork { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &ArrayD<f32>) -> ArrayD<f32> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backward(
        &mut self,
        grad_output: &ArrayD<f32>,
        momentum: f32,
        learning_rate: f32,
        epochs: usize,
    ) {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad, momentum, learning_rate, epochs);
        }
    }

    // 全批量梯度下降
    pub fn train(
        &mut self,
        inputs: &Array2<f32>,
        targets: &Array2<f32>,
        epochs: usize,
        momentum: f32,
        learning_rate: f32,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            // 前向传播
            let predictions = self.forward(&inputs.clone().into_dyn());

            // 计算损失 (均方误差)
            // loss = (1/4) * [(p1 - t1)^2 + (p2 - t2)^2 + (p3 -t3)^2 + (p4 - t4)^2]
            let diff = &predictions - &targets.clone().into_dyn();
            let loss = diff.mapv(|x| x.powi(2)).mean().unwrap();
            total_loss += loss;

            // 反向传播，对loss函数求梯度
            let grad_output = 2.0 * diff / inputs.shape()[0] as f32;
            self.backward(&grad_output, momentum, learning_rate, epoch);

            if epoch % 100 == 0 {
                println!("Epoch {}, Loss: {}", epoch, total_loss);
            }
        }
    }
}

/// 训练一个简单的XOR网络
fn main() {
    // 创建网络结构。整个网络的结构为: 2x4X1。
    // 输入层为2各神经元。只有一个隐藏层，有4个神经元。输出层只有一个神经元
    let mut network = NeuralNetwork::new();
    network.add_layer(DenseLayer::new(2, 4, Activation::ReLU)); // 隐藏层, 一次能够处理4组数据
    network.add_layer(DenseLayer::new(4, 1, Activation::Sigmoid)); // 输出层

    // 训练数据，一次批处理4组数据
    let inputs =
        Array::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();

    let targets = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

    // 训练网络
    network.train(&inputs, &targets, 3000, 0.9, 0.1);

    // 测试网络
    println!("\nTesting trained network:");
    for i in 0..4 {
        let input = inputs.slice(s![i, ..]).to_owned().into_dyn(); // 每次获取1行
        let input = input.to_shape((1, 2)).unwrap().to_owned().into_dyn();
        let output = network.forward(&input);
        println!("Input: {:?}, Output: {:?}\n", input, output);
    }
}
