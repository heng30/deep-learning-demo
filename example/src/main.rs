use anyhow::Result;
use mylib::cifar10::Cifar10Dataset;
use tch::{
    nn::{self, Module, Optimizer, OptimizerConfig, Sequential, VarStore},
    Device, IndexOp, Tensor,
};

const BATCH_SIZE: i64 = 32;
const MODEL_PATH: &str = "target/cifar10.tch";

#[tokio::main]
async fn main() -> Result<()> {
    // 设置随机种子以便结果可重现
    tch::manual_seed(0);

    let device = Device::cuda_if_available();

    let mut net = Cifar10Net::new(BATCH_SIZE, device).await?;

    net.train(MODEL_PATH)?;

    Ok(())
}

// 定义神经网络结构
#[derive(Debug)]
pub struct Cifar10Net {
    dataset: Cifar10Dataset,
    module: Sequential,
    device: Device,
    vs: VarStore,
    opt: Optimizer,
    batch_size: i64,
}

impl Cifar10Net {
    pub async fn new(batch_size: i64, device: Device) -> Result<Self> {
        let dataset = Cifar10Dataset::new(None, None).await?;
        let vs = nn::VarStore::new(device);
        let opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

        let p = &vs.root();
        let module = nn::seq()
            // Conv2d: input channels, output channels, kernel size, 默认padding=0和stride=1
            // 输出形状：[batch_size, 32, height - 2, width - 2]，则输出 [1, 32, 30, 30]）
            .add(nn::conv2d(p / "conv1", 3, 32, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            // 输出形状：[batch_size, 32, 15, 15]（30 / 2 = 15）
            .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
            // 输出形状：[batch_size, 64, 13, 13]（15 - 2 = 13）
            .add(nn::conv2d(p / "conv2", 32, 64, 3, Default::default()))
            .add_fn(|xs| xs.relu())
            // 输出形状：[batch_size, 64, 6, 6]（13 / 2 = 6，向下取整）
            .add_fn(|xs| xs.max_pool2d(2, 2, 0, 1, false))
            // 展开成一维输出
            .add_fn(|xs| xs.flatten(1, -1))
            // 全连接层
            .add(nn::linear(p / "fc1", 64 * 6 * 6, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "fc2", 512, 10, Default::default()));

        Ok(Cifar10Net {
            dataset,
            module,
            vs,
            device,
            opt,
            batch_size,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        self.module.forward(xs)
    }

    pub fn set_optimizer(&mut self, opt: nn::Optimizer) {
        self.opt = opt;
    }

    pub fn load_vs(&mut self, model_path: &str) -> Result<()> {
        self.vs.load(model_path)?;
        Ok(())
    }

    pub fn train(&mut self, save_path: impl AsRef<std::path::Path>) -> Result<()> {
        // label的值为0-9中的一个
        let train_images = self.dataset.train_images.to_device(self.device);
        let train_labels = self.dataset.train_labels.to_device(self.device);
        let train_images_size = train_images.size();
        let train_labels_size = train_labels.size();

        assert_eq!(train_images_size[0], train_labels_size[0]);

        // 创建CNN网络
        for epoch in 1..=1 {
            let mut correct = 0; // 正确数量
            let mut total_num = 0; // 累计总样本数量
            let mut total_loss = 0.0; // 累计总损失

            // TODO: 打乱数据

            for batch_index in 0..(train_images_size[0] / self.batch_size) {
                let start_index = batch_index * self.batch_size;
                let end_index = start_index + self.batch_size;

                // 获取训练和验证集
                let train_dataset = &train_images.i((start_index..end_index, ..));
                let vaild_dataset = &train_labels.i(start_index..end_index);

                let output = self.forward(train_dataset);

                // 计算交叉熵损失(在分类认为中常用), 会先对数据进行softmax，再进行叉熵计算
                let loss = output.cross_entropy_loss::<Tensor>(
                    &vaild_dataset,       // 类别索引标签
                    None,                 // 不设置权重
                    tch::Reduction::Mean, // 损失求平均
                    -1,                   // 忽略无效类别（默认）
                    0.,                   // label_smoothing（默认0）
                );

                // 清除梯度
                self.opt.zero_grad();

                // 方向传播并更新参数
                self.opt.backward_step(&loss);

                // 累计总样本数量
                total_num += self.batch_size;
                total_loss += loss.double_value(&[]) * self.batch_size as f64;

                // 求出最大概率类别的下标
                let predict = output.argmax(-1, true);

                // 获取正确的类别
                for index in 0..self.batch_size {
                    let index = index as i64;
                    if predict.int64_value(&[index]) == vaild_dataset.int64_value(&[index]) {
                        correct += 1;
                    }
                }

                if batch_index % 50 == 0 {
                    println!(
                        "Epoch: {:3}  Batch_index: {:3}  Loss: {:.6} Correct: {:.3}",
                        epoch,
                        batch_index,
                        total_loss / total_num as f64,
                        correct as f64 / total_num as f64
                    );
                }
            }
        }

        self.vs.save(save_path)?;

        Ok(())
    }

    pub async fn test(model_path: &str, batch_size: i64, device: Device) -> Result<()> {
        // 检查模型文件是否存在
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found at {}", model_path));
        }

        let dataset = Cifar10Dataset::new(None, None).await?;
        let test_images = dataset.test_images.to_device(device);
        let test_labels = dataset.test_labels.to_device(device);
        let test_images_size = test_images.size();
        let test_labels_size = test_labels.size();
        assert_eq!(test_images_size[0], test_labels_size[0]);

        let mut correct = 0;
        let mut net = Cifar10Net::new(batch_size, device).await?;

        // TODO:设置为评估模式, 不会更新权重、梯度等参数

        // 加载模型
        net.load_vs(model_path)?;

        for batch_index in 0..(test_images_size[0] / batch_size) {
            let start_index = batch_index * batch_size;
            let end_index = start_index + batch_size;

            // 获取训练和验证集
            let test_dataset = &test_images.i((start_index..end_index, ..));
            let vaild_dataset = &test_labels.i(start_index..end_index);

            // 运行模型
            let output = net.forward(&test_dataset);

            // 求出最大概率类别的下标
            let predict = output.argmax(-1, true);

            // 获取正确的类别
            for index in 0..BATCH_SIZE {
                let index = index as i64;
                if predict.int64_value(&[index]) == vaild_dataset.int64_value(&[index]) {
                    correct += 1;
                }
            }
        }

        println!(
            "Test Correct: {:.3}",
            correct as f64 / test_labels_size[0] as f64
        );

        Ok(())
    }
}
