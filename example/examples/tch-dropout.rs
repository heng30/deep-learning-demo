use tch::{Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);

    // 正常数据
    let mut w = Tensor::randn([15, 1], (Kind::Float, Device::Cpu)).requires_grad_(true);

    let x = Tensor::randint(10, [5, 15], (Kind::Int64, Device::Cpu)).to_kind(Kind::Float);

    x.matmul(&w).sum(Kind::Float).backward();

    println!(
        "Gradient:\n{:?}",
        w.grad()
            .reshape([1, -1])
            .squeeze()
            .iter::<f64>()?
            .collect::<Vec<_>>()
    );

    // Dropout
    w.zero_grad();
    let x = x.dropout(0.8, true);
    x.matmul(&w).sum(Kind::Float).backward();

    println!(
        "\nDropout Gradient:\n{:?}",
        w.grad()
            .reshape([1, -1])
            .squeeze()
            .iter::<f64>()?
            .collect::<Vec<_>>()
    );

    Ok(())
}
