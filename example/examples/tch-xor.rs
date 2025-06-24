use tch::{Device, Tensor, nn, nn::Module, nn::OptimizerConfig};

fn main() {
    // 设置随机种子以便结果可重现
    tch::manual_seed(42);

    // 定义 XOR 问题的输入和输出
    let inputs = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .reshape(&[4, 2])
        .to_device(Device::cuda_if_available());

    let targets = Tensor::from_slice(&[0.0_f32, 1.0, 1.0, 0.0])
        .reshape(&[4, 1])
        .to_device(Device::cuda_if_available());

    // 创建神经网络
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());

    // 创建优化器
    let mut opt = nn::Adam::default().build(&vs, 1e-1).unwrap();

    // 训练循环
    for epoch in 1..=1000 {
        let output = net.forward(&inputs);

        // 计算损失 (MSE)
        let loss = output.mse_loss(&targets, tch::Reduction::Mean);

        opt.backward_step(&loss);

        if epoch % 100 == 0 {
            println!("Epoch: {:4} Loss: {:.6}", epoch, loss.double_value(&[]));
        }
    }

    // 测试训练好的模型
    let test_output = net.forward(&inputs);
    println!("\nTest results:\n");
    println!("Inputs:\n{}\n", inputs);
    println!("Predictions:\n{}\n", test_output);
    println!("Targets:\n{}", targets);
}

// 定义神经网络结构
#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Self {
        let fc1 = nn::linear(vs, 2, 2, Default::default()); // 输入层到隐藏层
        let fc2 = nn::linear(vs, 2, 1, Default::default()); // 隐藏层到输出层

        Net { fc1, fc2 }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // 第一层 + ReLU激活函数
        let xs = self.fc1.forward(xs).relu();

        // 第二层 + Sigmoid激活函数 (将输出压缩到0-1之间)
        self.fc2.forward(&xs).sigmoid()
    }
}
