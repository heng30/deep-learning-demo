use anyhow::Result;
use mylib::cifar10::Cifar10Dataset;
use tch::{
    nn::{self, Module, OptimizerConfig},
    Device, IndexOp, Tensor,
};

const BATCH_SIZE: i64 = 32;
const MODEL_PATH: &str = "target/cifar10.tch";

#[tokio::main]
async fn main() -> Result<()> {
    // 设置随机种子以便结果可重现
    tch::manual_seed(0);

    let cifar10 = Cifar10Dataset::new(None, None).await?;
    println!("{:?}", cifar10);

    // label的值为0-9中的一个
    let device = Device::cuda_if_available();
    let train_images = cifar10.train_images.to_device(device);
    let train_labels = cifar10.train_labels.to_device(device);
    let test_images = cifar10.test_images.to_device(device);
    let test_labels = cifar10.test_labels.to_device(device);

    train(train_images, train_labels, device)?;
    test(test_images, test_labels, device)?;

    Ok(())
}

fn train(train_images: Tensor, train_labels: Tensor, device: Device) -> Result<()> {
    let train_images_size = train_images.size();
    let train_labels_size = train_labels.size();

    assert_eq!(train_images_size[0], train_labels_size[0]);

    // 创建CNN网络
    let vs = nn::VarStore::new(device);
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    for epoch in 1..=1 {
        let mut correct = 0; // 正确数量
        let mut total_num = 0; // 累计总样本数量
        let mut total_loss = 0.0; // 累计总损失

        // TODO: 打乱数据

        for batch_index in 0..(train_images_size[0] / BATCH_SIZE) {
            let start_index = batch_index * BATCH_SIZE;
            let end_index = start_index + BATCH_SIZE;

            // 获取训练和验证集
            let train_dataset = &train_images.i((start_index..end_index, ..));
            let vaild_dataset = &train_labels.i(start_index..end_index);

            let output = net.forword(train_dataset);

            // 计算交叉熵损失(在分类认为中常用), 会先对数据进行softmax，再进行叉熵计算
            let loss = output.cross_entropy_loss::<Tensor>(
                &vaild_dataset,       // 类别索引标签
                None,                 // 不设置权重
                tch::Reduction::Mean, // 损失求平均
                -1,                   // 忽略无效类别（默认）
                0.,                   // label_smoothing（默认0）
            );

            // 清除梯度
            opt.zero_grad();

            // 方向传播并更新参数
            opt.backward_step(&loss);

            // 累计总样本数量
            total_num += BATCH_SIZE;
            total_loss += loss.double_value(&[]) * BATCH_SIZE as f64;

            // 求出最大概率类别的下标
            let predict = output.argmax(-1, true);

            // 获取正确的类别
            for index in 0..BATCH_SIZE {
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

    vs.save(MODEL_PATH)?;

    Ok(())
}

fn test(test_images: Tensor, test_labels: Tensor, device: Device) -> Result<()> {
    let test_images_size = test_images.size();
    let test_labels_size = test_labels.size();
    assert_eq!(test_images_size[0], test_labels_size[0]);

    // let mut correct = 0;
    // let mut vs = nn::VarStore::new(Device::cuda_if_available());
    //
    // // 获取维度
    // let features_dim = phone_price.features.first().unwrap().len() as i64;
    //
    // let labels_dim = phone_price
    //     .labels
    //     .iter()
    //     .copied()
    //     .collect::<HashSet<i64>>()
    //     .len() as i64;
    //
    // let net = Net::new(&vs.root(), features_dim, labels_dim);
    //
    // // 检查模型文件是否存在
    // if !std::path::Path::new(MODEL_PATH).exists() {
    //     return Err(anyhow::anyhow!("Model file not found at {}", MODEL_PATH));
    // }
    //
    // vs.load(MODEL_PATH)?;
    //
    // // 构建测试数据
    // let test_index = phone_price.features.len() - TEST_COUNTS;
    //
    // for batch_index in 0..(TEST_COUNTS / BATCH_SIZE) {
    //     let start_index = test_index + batch_index * BATCH_SIZE;
    //     let end_index = start_index + BATCH_SIZE;
    //
    //     let test_dataset = Tensor::from_slice2(&phone_price.features[start_index..end_index])
    //         .to_kind(Kind::Float)
    //         .to_device(Device::cuda_if_available());
    //
    //     let vaild_dataset = Tensor::from_slice(&phone_price.labels[start_index..end_index])
    //         .to_kind(Kind::Int64)
    //         .to_device(Device::cuda_if_available());
    //
    //     // 运行模型
    //     let output = net.forward(&test_dataset);
    //
    //     // 求出最大概率类别的下标
    //     let predict = output.argmax(-1, true);
    //
    //     // 获取正确的类别
    //     for index in 0..BATCH_SIZE {
    //         let index = index as i64;
    //         if predict.int64_value(&[index]) == vaild_dataset.int64_value(&[index]) {
    //             correct += 1;
    //         }
    //     }
    // }
    //
    // println!("Test Correct: {:.3}", correct as f64 / TEST_COUNTS as f64);
    //

    Ok(())
}

// 定义神经网络结构
#[derive(Debug)]
struct Net {
    module: nn::Sequential,
}

impl Net {
    fn new(p: &nn::Path) -> Self {
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

        Net { module }
    }

    fn forword(&self, xs: &Tensor) -> Tensor {
        self.module.forward(xs)
    }
}
