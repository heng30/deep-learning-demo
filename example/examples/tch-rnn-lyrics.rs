use anyhow::Result;
use jieba_rs::Jieba;
use regex::Regex;
use std::collections::{BTreeSet, HashMap};
use tch::{
    nn::{self, Embedding, Linear, ModuleT, Optimizer, OptimizerConfig, VarStore, LSTM, RNN},
    Device, Kind, Tensor,
};

const SEQ_LEN: usize = 4;
const TRAIN_EPOCH: usize = 100;
const HIDDEN_DIM: i64 = 256;
const EMBEDDING_DIM: i64 = 128;
const DATA_PATH: &str = "data/jaychou_lyrics.txt";
const MODEL_PATH: &str = "target/jaychou_lyrics.safetensers";

fn main() -> Result<()> {
    // 设置随机种子以便结果可重现
    tch::manual_seed(0);
    let device = Device::cuda_if_available();

    // 读取数据
    let sentence = std::fs::read_to_string(DATA_PATH)?;
    println!("Sentence len: {:?}", sentence.as_str().chars().count());

    // 构建词表
    let vocab = Vocabulary::new(&sentence);
    println!("Vocabulary len: {}", vocab.len());
    println!("Tokens len: {}", vocab.tokens().len());

    let mut net = Net::new(vocab.len() as i64, EMBEDDING_DIM, HIDDEN_DIM, device);
    net.train(vocab.clone(), SEQ_LEN, TRAIN_EPOCH, MODEL_PATH)?;

    Net::predict(
        "我",
        20,
        vocab,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        MODEL_PATH,
        device,
    )?;

    Ok(())
}

// 词表
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<String>, // 保存结巴分词后的词组
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    next_id: usize, // 记录下一个可以添加的下标
    unknown_token: String,
    padding_token: String,
}

// 词汇表
impl Vocabulary {
    pub fn new(sentence: &str) -> Self {
        let mut vocab = Vocabulary {
            tokens: Vec::new(),
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            next_id: 0,
            unknown_token: "<unk>".to_string(), // 未知词标记
            padding_token: "<pad>".to_string(), // 填充标记
        };
        vocab.add_word(&vocab.unknown_token.clone()); // 添加未知词标记
        vocab.add_word(&vocab.padding_token.clone()); // 添加填充标记

        let sentence = Vocabulary::clean_text(sentence);
        let tokens = Vocabulary::tokenize_sentence(&sentence);
        vocab.tokens = tokens.clone();
        vocab.add_words(tokens);

        vocab
    }

    // 对单词去重
    pub fn unique_words(tokens: Vec<String>) -> Vec<String> {
        let words: BTreeSet<String> = tokens.into_iter().collect();
        let mut words = words.into_iter().collect::<Vec<String>>();
        words.sort();
        words
    }

    // 将词和下标添加到对应的词表中
    pub fn add_word(&mut self, word: &str) -> usize {
        if let Some(&id) = self.word_to_id.get(word) {
            id
        } else {
            let id = self.next_id;
            self.word_to_id.insert(word.to_string(), id);
            self.id_to_word.push(word.to_string());
            self.next_id += 1;
            id
        }
    }

    // 添加词
    pub fn add_words(&mut self, words: Vec<String>) {
        let words = Vocabulary::unique_words(words);
        for word in words {
            self.add_word(&word);
        }
    }

    // 获取词对应的下标
    pub fn get_id(&self, word: &str) -> usize {
        *self.word_to_id.get(word).unwrap_or_else(|| {
            self.word_to_id
                .get(&self.unknown_token)
                .expect("Unknown token not in vocabulary")
        })
    }

    // 获取下标对应的词
    pub fn get_word(&self, id: usize) -> Option<&str> {
        self.id_to_word.get(id).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.next_id
    }

    pub fn is_empty(&self) -> bool {
        self.next_id == 0
    }

    // 分词
    fn tokenize_sentence(sentence: &str) -> Vec<String> {
        let jieba = Jieba::new();
        jieba
            .cut(sentence, false)
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // 清理数据
    pub fn clean_text(all_chars: &str) -> String {
        // 替换换行符和回车符为空格
        let replaced_newlines = all_chars.replace(['\n', '\r'], " ");

        // 创建正则表达式来匹配非中文字符
        let re = Regex::new(
            r"[A-Za-z0-9\.\*\+\?\]\[＞＜<】〇〗〖\\\\【>!?>><<~/\u3000》,☆。！《》、`,～？…]",
        )
        .unwrap();

        // 移除非中文字符
        let chinese_only = re.replace_all(&replaced_newlines, "");

        // 取前N个字符
        let training_set: String = chinese_only.chars().collect();

        training_set
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

// 定义神经网络结构
#[allow(unused)]
#[derive(Debug)]
pub struct Net {
    embedding: Embedding,
    lstm: LSTM,
    output: Linear,

    device: Device,
    vs: VarStore,
    opt: Optimizer,
}

impl Net {
    pub fn new(vocab_size: i64, embedding_dim: i64, hidden_dim: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();
        let opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

        // 词嵌入层
        let embedding = nn::embedding(
            p / "embedding",
            vocab_size,
            embedding_dim,
            Default::default(),
        );

        // 循环神经网络层
        let lstm = nn::lstm(p / "lstm", embedding_dim, hidden_dim, Default::default());

        // 输出层
        let output = nn::linear(p / "output", hidden_dim, vocab_size, Default::default());

        Net {
            embedding,
            lstm,
            output,
            vs,
            device,
            opt,
        }
    }

    pub fn load_vs(&mut self, model_path: &str) -> Result<()> {
        self.vs.load(model_path)?;
        Ok(())
    }

    // batch_size: 输入多少个句子
    // seq_len: 每个句子的长度
    // features: 每个词的维度
    pub fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // 输入形状：[seq_len, word_id]
        // 输出形状：[seq_len, features]
        let xs = self.embedding.forward_t(xs, train);

        // 正则化
        let xs = xs.dropout(0.2, train);

        // println!("embedding: {:?}", xs.size());

        // 添加batch_size
        let xs = xs.unsqueeze(0);

        // 输入形状 [batch_size, seq_len, features].
        let (xs, _hidden) = self.lstm.seq(&xs);

        // 移除batch_size
        let xs = xs.squeeze();

        // println!("lstm: {:?}", xs.size());

        // 正则化
        let xs = xs.dropout(0.2, train);

        // 线性输出层
        let output = self.output.forward_t(&xs, train);

        output
    }

    pub fn train(
        &mut self,
        vocab: Vocabulary,
        seq_len: usize,
        epoch: usize,
        save_path: &str,
    ) -> Result<()> {
        let mut seq_len = seq_len;

        for epoch in 1..=epoch {
            let mut correct = 0; // 正确数量
            let mut total_num = 0; // 累计总样本数量
            let mut total_loss = 0.0; // 累计总损失

            let mut train_set = vec![];
            let mut label_set = vec![];

            // 每一个epoch更新句子长度
            seq_len = seq_len + 1;

            // 获取训练集合和验证集
            for items in vocab.tokens().chunks(seq_len + 1) {
                if items.len() < seq_len + 1 {
                    break;
                }

                let items: Vec<_> = items.iter().map(|item| vocab.get_id(item) as i64).collect();

                train_set.push(Tensor::from_slice(&items[..seq_len]).to_device(self.device));
                label_set.push(Tensor::from(&items[1..seq_len + 1]).to_device(self.device));
            }

            //  打乱数据
            suffle_tensor_rows(&mut train_set, &mut label_set)?;

            for (batch_index, (train_item, label_item)) in
                train_set.iter().zip(label_set.iter()).enumerate()
            {
                let output = self.forward_t(train_item, true);

                // 计算交叉熵损失(在分类认为中常用), 会先对数据进行softmax，再进行叉熵计算
                let loss = output.cross_entropy_loss::<Tensor>(
                    label_item,           // 类别索引标签
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
                total_num += seq_len;
                total_loss += loss.double_value(&[]) * seq_len as f64;

                // 求出最大概率类别的下标
                let predict = output.argmax(-1, true);

                // 获取正确的类别
                for index in 0..seq_len {
                    let index = index as i64;
                    if predict.int64_value(&[index]) == label_item.int64_value(&[index]) {
                        correct += 1;
                    }
                }

                if batch_index % 50 == 0 {
                    println!(
                        "Epoch: {:2}  Batch_index: {:4}  Loss: {:.6} Correct: {:.3}",
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

    pub fn predict(
        init_word: &str,
        preict_count: usize,
        vocab: Vocabulary,
        embedding_dim: i64,
        hidden_dim: i64,
        model_path: &str,
        device: Device,
    ) -> Result<()> {
        // 检查模型文件是否存在
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!("Model file not found at {}", model_path));
        }

        let mut net = Net::new(vocab.len() as i64, embedding_dim, hidden_dim, device);

        // 加载模型
        net.load_vs(model_path)?;

        let mut input = vec![init_word];

        for _ in 0..preict_count {
            let items: Vec<_> = input.iter().map(|item| vocab.get_id(item) as i64).collect();
            let xs = Tensor::from_slice(&items[..]).to_device(device);

            let output = net.forward_t(&xs, false);

            // 获取最后时间步输出，求出最大概率类别的下标
            let last_output = output.select(0, -1);
            let predict_index = last_output.argmax(-1, false).int64_value(&[]);

            if let Some(word) = vocab.get_word(predict_index as usize) {
                input.push(word);
                println!(
                    "{}",
                    input.iter().map(|s| s.to_string()).collect::<String>()
                );
            }
        }

        Ok(())
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
