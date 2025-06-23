use anyhow::{Context, Result};
use csv::Reader;
use std::collections::HashSet;
use tch::{
    Device, Kind, Tensor,
    nn::{self, Module, OptimizerConfig},
};

const BATCH_SIZE: usize = 8;
const VALIDATE_COUNTS: usize = BATCH_SIZE * 10; // 使用80个元素进行验证
const TEST_COUNTS: usize = BATCH_SIZE * 10; // 使用80个元素进行测试
const DATA_PATH: &str = "data/cell-phone-price-prediction.csv";
const MODEL_PATH: &str = "target/cell-phone-price.pth";

fn main() -> Result<()> {
    let rand_rows = Tensor::randint(2, [1, 30], (Kind::Int64, Device::Cpu));

    println!("{:?}", rand_rows.print());

    // tch::manual_seed(0);
    //
    // let phone_price = load_data(DATA_PATH)?;
    //
    // train(&phone_price)?;
    // test(&phone_price)?;

    Ok(())
}

fn train(phone_price: &PhonePrice) -> Result<()> {
    // 获取维度
    let features_dim = phone_price.features.first().unwrap().len() as i64;

    let labels_dim = phone_price
        .labels
        .iter()
        .copied()
        .collect::<HashSet<i64>>()
        .len() as i64;

    // ==================== 训练神经网络 ================
    let train_counts = phone_price.features.len() - VALIDATE_COUNTS - TEST_COUNTS;

    let mut features = phone_price.features[..train_counts]
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    let mut labels = phone_price.labels[..train_counts]
        .iter()
        .cloned()
        .collect::<Vec<_>>();

    let vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root(), features_dim, labels_dim);
    let mut opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

    // 每轮验证模型，避免过拟合
    let mut best_validate_acc = 0.0;
    let mut no_improve_epochs = 0;

    // 训练循环
    for epoch in 1..=50 {
        let mut correct = 0; // 正确数量
        let mut total_num = 0; // 累计总样本数量
        let mut total_loss = 0.0; // 累计总损失

        // 打乱训练数据数据
        suffle_tensor_rows(&mut features[..train_counts], &mut labels[..train_counts])?;

        // 按批次进行训练
        for batch_index in 0..(train_counts / BATCH_SIZE) {
            let start_index = batch_index * BATCH_SIZE;
            let end_index = start_index + BATCH_SIZE;

            // 训练集
            let train_dataset = Tensor::from_slice2(&features[start_index..end_index])
                .to_kind(Kind::Float)
                .to_device(Device::cuda_if_available());

            // 添加高斯噪声
            let train_dataset = &train_dataset + &Tensor::randn_like(&train_dataset) * 0.01;

            // 验证集
            let vaild_dataset = Tensor::from_slice(&labels[start_index..end_index])
                .to_kind(Kind::Int64)
                .to_device(Device::cuda_if_available());

            let output = net.forward(&train_dataset);

            // 计算平均损失, 会先对数据进行softmax，再进行叉熵计算
            let loss = output.cross_entropy_loss::<Tensor>(
                &vaild_dataset,       // 类别索引标签
                None,                 // 不设置权重
                tch::Reduction::Mean, // 损失求平均
                -1,                   // 忽略无效类别（默认）
                0.,                   // label_smoothing（默认0）
            );

            // 梯度清零
            opt.zero_grad();

            // 方向传播并更新参数
            opt.backward_step(&loss);

            // 累计总样本数量
            total_num += BATCH_SIZE;

            // 累计总损失
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

        // 每个epoch结束后在验证集上测试
        let validate_acc = validate(&net, &phone_price);
        println!("Epoch {} validate accuracy: {:.3}", epoch, validate_acc);

        // 早停逻辑
        if validate_acc > best_validate_acc {
            best_validate_acc = validate_acc;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs += 1;
            if no_improve_epochs >= 10 {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    vs.save(MODEL_PATH)?;

    Ok(())
}

// 训练验证
fn validate(net: &Net, phone_price: &PhonePrice) -> f64 {
    let mut correct = 0;

    // 构建测试数据
    let validate_index = phone_price.features.len() - VALIDATE_COUNTS - TEST_COUNTS;

    for batch_index in 0..(VALIDATE_COUNTS / BATCH_SIZE) {
        let start_index = validate_index + batch_index * BATCH_SIZE;
        let end_index = start_index + BATCH_SIZE;

        let test_dataset = Tensor::from_slice2(&phone_price.features[start_index..end_index])
            .to_kind(Kind::Float)
            .to_device(Device::cuda_if_available());

        let vaild_dataset = Tensor::from_slice(&phone_price.labels[start_index..end_index])
            .to_kind(Kind::Int64)
            .to_device(Device::cuda_if_available());

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

    correct as f64 / TEST_COUNTS as f64
}

// 测试
fn test(phone_price: &PhonePrice) -> Result<()> {
    let mut correct = 0;
    let mut vs = nn::VarStore::new(Device::cuda_if_available());

    // 获取维度
    let features_dim = phone_price.features.first().unwrap().len() as i64;

    let labels_dim = phone_price
        .labels
        .iter()
        .copied()
        .collect::<HashSet<i64>>()
        .len() as i64;

    let net = Net::new(&vs.root(), features_dim, labels_dim);

    // 检查模型文件是否存在
    if !std::path::Path::new(MODEL_PATH).exists() {
        return Err(anyhow::anyhow!("Model file not found at {}", MODEL_PATH));
    }

    vs.load(MODEL_PATH)?;

    // 构建测试数据
    let test_index = phone_price.features.len() - TEST_COUNTS;

    for batch_index in 0..(TEST_COUNTS / BATCH_SIZE) {
        let start_index = test_index + batch_index * BATCH_SIZE;
        let end_index = start_index + BATCH_SIZE;

        let test_dataset = Tensor::from_slice2(&phone_price.features[start_index..end_index])
            .to_kind(Kind::Float)
            .to_device(Device::cuda_if_available());

        let vaild_dataset = Tensor::from_slice(&phone_price.labels[start_index..end_index])
            .to_kind(Kind::Int64)
            .to_device(Device::cuda_if_available());

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

    println!("Test Correct: {:.3}", correct as f64 / TEST_COUNTS as f64);

    Ok(())
}

#[derive(Debug, Clone)]
struct PhonePrice {
    #[allow(unused)]
    headers: Vec<String>,
    features: Vec<Vec<f64>>,
    labels: Vec<i64>,
}

fn load_data<A: AsRef<std::path::Path>>(path: A) -> Result<PhonePrice> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut rdr = Reader::from_reader(file);

    // 获取头信息（可选）
    let headers = rdr.headers()?.iter().map(String::from).collect::<Vec<_>>();

    // 用于存储特征和标签
    let mut features = Vec::new();
    let mut labels = Vec::new();

    // 记录第一行的元素数量，用于后续检查
    let mut expected_len = 0;

    for (i, result) in rdr.records().enumerate() {
        let record = result?;

        // 如果是第一行，设置期望的长度
        if i == 0 {
            expected_len = record.len();
        }

        // 检查每行的元素数量是否一致
        if record.len() != expected_len {
            anyhow::bail!(format!(
                "Line {} has {} elements, expected {}",
                i + 1,
                record.len(),
                expected_len
            ));
        }

        // 分割特征和标签
        // 特征是从第一个到倒数第二个元素
        let feature_values: Vec<f64> = record
            .iter()
            .take(record.len() - 1)
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();

        // 标签是最后一个元素
        let label = record
            .iter()
            .last()
            .with_context(|| "Empty record")?
            .parse()
            .unwrap_or(0);

        features.push(feature_values);
        labels.push(label);
    }

    // 计算每个特征的均值和标准差
    let mut means = vec![0.0; features[0].len()];
    let mut stds = vec![0.0; features[0].len()];

    // 计算均值
    for feature in &features {
        for (i, &val) in feature.iter().enumerate() {
            means[i] += val;
        }
    }
    means.iter_mut().for_each(|m| *m /= features.len() as f64);

    // 计算标准差
    for feature in &features {
        for (i, &val) in feature.iter().enumerate() {
            stds[i] += (val - means[i]).powi(2);
        }
    }
    stds.iter_mut().for_each(|s| {
        *s = (*s / features.len() as f64).sqrt();
        if *s == 0.0 {
            *s = 1.0
        }
    });

    // 标准化特征
    for feature in &mut features {
        for (i, val) in feature.iter_mut().enumerate() {
            *val = (*val - means[i]) / stds[i];
        }
    }

    Ok(PhonePrice {
        headers,
        features,
        labels,
    })
}

// 定义神经网络结构
#[derive(Debug)]
struct Net {
    input_layer: nn::Linear,
    hidden_layers: Vec<nn::Linear>,
    output_layer: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path, features_dim: i64, labels_dim: i64) -> Self {
        let input_layer = nn::linear(vs, features_dim, 128, Default::default()); // 输入层到隐藏层

        let mut hidden_layers = vec![];
        hidden_layers.push(nn::linear(vs, 128, 256, Default::default()));
        hidden_layers.push(nn::linear(vs, 256, 512, Default::default()));
        hidden_layers.push(nn::linear(vs, 512, 128, Default::default()));

        let output_layer = nn::linear(vs, 128, labels_dim, Default::default()); // 隐藏层到输出层

        Net {
            input_layer,
            hidden_layers,
            output_layer,
        }
    }
}

impl Module for Net {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = self.input_layer.forward(xs).relu().dropout(0.1, true);

        for layer in &self.hidden_layers {
            xs = layer.forward(&xs).relu().dropout(0.2, true);
        }

        // 因为cross_entropy_loss已经对数据进行softmax，这里就不需要使用`sigmoid`函数了
        self.output_layer.forward(&xs)
    }
}

fn suffle_tensor_rows<T, U>(train_set: &mut [T], valid_set: &mut [U]) -> Result<()> {
    assert_eq!(train_set.len(), valid_set.len());

    let rand_rows = Tensor::randint(
        train_set.len() as i64,
        [1, train_set.len() as i64],
        (Kind::Int64, Device::Cpu),
    );

    // 打乱数据
    for (index, item) in rand_rows.squeeze().iter::<i64>()?.enumerate() {
        let item = item as usize;

        if index == item {
            continue;
        }

        train_set.swap(index, item);
        valid_set.swap(index, item);
    }

    Ok(())
}
