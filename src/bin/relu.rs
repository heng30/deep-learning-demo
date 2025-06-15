use plotters::prelude::*;

// ReLU 函数
fn relu(x: f64) -> f64 {
    x.max(0.)
}

// ReLU 函数的导数
fn relu_derivative(x: f64) -> f64 {
    if x <= 0. { 0. } else { 1. }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建图像文件
    let root = BitMapBackend::new("target/output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 设置图表
    let mut chart = ChartBuilder::on(&root)
        .caption("ReLU Function and its Derivative", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-2.0..2.0, 0.0..2.0)?;

    // 配置网格和坐标轴
    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // 绘制 ReLU 函数
    chart
        .draw_series(LineSeries::new(
            (-100..=100).map(|x| x as f64 / 50.0).map(|x| (x, relu(x))),
            &RED,
        ))?
        .label("ReLu")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 绘制 ReLU 导数
    chart
        .draw_series(LineSeries::new(
            (-100..=100)
                .map(|x| x as f64 / 50.0)
                .map(|x| (x, relu_derivative(x))),
            &BLUE,
        ))?
        .label("Derivative")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // 添加图例
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
