use plotters::prelude::*;
use std::f64::consts::E;

fn softmax(z: &[f64]) -> Vec<f64> {
    let exp_z: Vec<f64> = z.iter().map(|&x| E.powf(x)).collect();
    let sum_exp_z: f64 = exp_z.iter().sum();
    exp_z.iter().map(|&x| x / sum_exp_z).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 创建图像文件
    let root = BitMapBackend::new("target/output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Softmax Function Probability Distribution",
            ("sans-serif", 40),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0..4.0, 0.0..1.0)?;

    // 绘制网格
    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    // 计算softmax概率分布
    let z = vec![0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.08, 1.1, 0.09, 3.75];
    let p_z = softmax(&z);

    chart
        .draw_series(
            z.iter()
                .zip(p_z.iter())
                .map(|(&x, &y)| Circle::new((x, y), 5, RED.filled())),
        )?
        .label("Probability")
        .legend(|(x, y)| Circle::new((x, y), 5, RED.filled()));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
