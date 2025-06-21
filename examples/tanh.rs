use plotters::prelude::*;

// Tanh 函数
fn tanh(x: f64) -> f64 {
    x.tanh()
}

// Tanh 函数的导数
fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建图像文件
    let root = BitMapBackend::new("target/output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 设置图表
    let mut chart = ChartBuilder::on(&root)
        .caption("Tanh Function and its Derivative", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-3.0..3.0, -1.2..1.2)?;

    // 配置网格和坐标轴
    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // 绘制 Tanh 函数
    chart
        .draw_series(LineSeries::new(
            (-60..=60).map(|x| x as f64 / 20.0).map(|x| (x, tanh(x))),
            &RED,
        ))?
        .label("Tanh")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // 绘制 Tanh 导数
    chart
        .draw_series(LineSeries::new(
            (-60..=60)
                .map(|x| x as f64 / 20.0)
                .map(|x| (x, tanh_derivative(x))),
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
