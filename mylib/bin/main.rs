use anyhow::Result;
use mylib::cifar10::Cifar10Dataset;

#[tokio::main]
async fn main() -> Result<()> {
    let dataset = Cifar10Dataset::new(None, None).await?;
    println!("{:?}", dataset);

    Ok(())
}
