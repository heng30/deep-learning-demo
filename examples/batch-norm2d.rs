use tch::{
    Device, Kind, Tensor,
    nn::{self, ModuleT},
};

// 使用`batch_norm2d`进行输入数据批量归一化。
// 为了避免数据因为批次差距过大，在训练的时候造成参数变化过于剧烈。
// 使用`batch_norm2d`处理输入数据后，能有效避免这种情况。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);

    // [batch_size, channel_size, height(row), width(column)]
    let input = Tensor::randint(10, [2, 2, 3, 3], (Kind::Int64, Device::Cpu)).to_kind(Kind::Float);

    println!("input:\n{:?}", input.print());
    println!("\n=======================\n");

    let config = nn::BatchNormConfig {
        cudnn_enabled: true,
        eps: 1e-6,
        momentum: 0.1,
        affine: false,
        ws_init: nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
        bs_init: nn::Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    };
    let vs = nn::VarStore::new(Device::Cpu);
    let bn = nn::batch_norm2d(&vs.root(), 2, config);
    let output = bn.forward_t(&input, true);
    println!("output:\n{:?}", output.print());

    Ok(())
}
