use anyhow::Result;
use jieba_rs::Jieba;
use std::collections::{BTreeSet, HashMap};
use tch::{
    nn::{self, ModuleT, Optimizer, OptimizerConfig, Sequential, VarStore, RNN},
    Device, Tensor,
};

fn main() -> Result<()> {
    // 设置随机种子以便结果可重现
    tch::manual_seed(0);
    let device = Device::cuda_if_available();

    let sentence = "北京冬奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归程。";
    let tokens = tokenize_sentence(sentence);
    println!("Tokens: {:?}", tokens);

    let mut vocab = Vocabulary::new();
    vocab.add_words(tokens.clone());

    println!("Vocabulary size: {}", vocab.len());
    println!("'北京' ID: {}", vocab.get_id("北京"));
    println!("'地球' ID (unknown): {}", vocab.get_id("地球")); // 应该返回<unk>的ID
    println!("ID 0 word: {:?}", vocab.get_word(0)); // 应该返回<unk>

    // 1. 定义词汇表大小和嵌入维度
    let vocab_size = vocab.len();
    let embedding_dim = 4; // 为了观察方便使用4。在具体训练中应该取一个比较大的值：128

    // 3. 构建嵌入层网络
    let embedding_module = EmbeddingModule::new(vocab_size as i64, embedding_dim, device);

    let mut embedding_vetors = vec![];
    println!("\n============= Embedding Vector=============");
    for word in tokens.iter() {
        let idx = vocab.get_id(word) as i64;
        let xs = Tensor::from(idx);
        let output = embedding_module.forward_t(&xs, true);

        println!("{:?}", output);
        embedding_vetors.push(output);
    }

    println!("\n============= LSTM Vector=============");
    let hidden_size = 2; // 为了观察方便使用2。在具体训练中应该quo一个比较大的值：256
    let lstm_module = LstmModule::new(embedding_dim, hidden_size, device);
    for v in embedding_vetors.iter() {
        // 形状由[4] -> [1, 1, 4]
        let input = v.unsqueeze(0).unsqueeze(0);

        let (output, state) = lstm_module.forward(&input);
        println!("output size: {:?}", output.size());
        println!("state h size: {:?}", state.h().size());
        println!("state c size: {:?}", state.c().size());

        println!(
            "output: {:?}\n",
            output
                .reshape([1, -1])
                .squeeze()
                .iter::<f64>()?
                .collect::<Vec<_>>()
        );
    }

    Ok(())
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

// 词表
pub struct Vocabulary {
    word_to_id: HashMap<String, usize>,
    id_to_word: Vec<String>,
    next_id: usize, // 记录下一个可以添加的下标
    unknown_token: String,
    padding_token: String,
}

// 词汇表
impl Vocabulary {
    pub fn new() -> Self {
        let mut vocab = Vocabulary {
            word_to_id: HashMap::new(),
            id_to_word: Vec::new(),
            next_id: 0,
            unknown_token: "<unk>".to_string(), // 未知词标记
            padding_token: "<pad>".to_string(), // 填充标记
        };
        vocab.add_word(&vocab.unknown_token.clone()); // 添加未知词标记
        vocab.add_word(&vocab.padding_token.clone()); // 添加填充标记
        vocab
    }

    // 对单词去重
    pub fn unique_words(tokens: Vec<String>) -> Vec<String> {
        let words: BTreeSet<String> = tokens.into_iter().collect();
        words.into_iter().collect::<Vec<String>>()
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
}

// 定义神经网络结构
#[allow(unused)]
#[derive(Debug)]
pub struct EmbeddingModule {
    module: Sequential,
    device: Device,
    vs: VarStore,
    opt: Optimizer,
}

impl EmbeddingModule {
    pub fn new(vocab_size: i64, embedding_dim: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let opt = nn::Adam::default().build(&vs, 1e-4).unwrap();

        let p = &vs.root();
        let module = nn::seq().add(nn::embedding(
            p / "embedding",
            vocab_size,
            embedding_dim,
            Default::default(),
        ));

        EmbeddingModule {
            module,
            vs,
            device,
            opt,
        }
    }

    pub fn forward_t(&self, xs: &Tensor, is_train: bool) -> Tensor {
        self.module.forward_t(xs, is_train)
    }
}

#[allow(unused)]
#[derive(Debug)]
pub struct LstmModule {
    vs: VarStore,
    lstm: nn::LSTM,
    device: Device,
}

impl LstmModule {
    pub fn new(in_dim: i64, hidden_dim: i64, device: Device) -> Self {
        let vs = nn::VarStore::new(device);
        let p = &vs.root();

        let lstm = nn::lstm(p / "lstm", in_dim, hidden_dim, Default::default());
        LstmModule { vs, lstm, device }
    }

    // 输入形状 [batch_size, seq_len, features].
    // batch_size: 输入多少个句子
    // seq_len: 每个句子的长度
    // features: 每个词的维度
    pub fn forward(&self, xs: &Tensor) -> (Tensor, nn::LSTMState) {
        self.lstm.seq(xs)
    }
}
