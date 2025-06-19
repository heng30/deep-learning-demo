use plotters::prelude::*;
use tch::{Device, Kind, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);
    let temperature = Tensor::randn([30], (Kind::Float, Device::Cpu)).abs() * 10.;

    let mut exp_weight_avg = vec![];
    let beta = 0.6;

    // 求加权平均数
    for (index, temp) in temperature.iter::<f64>()?.enumerate() {
        if index == 0 {
            exp_weight_avg.push(temp);
            continue;
        }

        let new_temp = exp_weight_avg.last().unwrap() * beta + (1. - beta) * temp;
        exp_weight_avg.push(new_temp);
    }

    // println!("{exp_weight_avg:?}");

    // 创建图像文件
    let root = BitMapBackend::new("target/output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // 设置图表
    let mut chart = ChartBuilder::on(&root)
        .caption("Weight mean", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(1.0..30.0, 0.0..20.0)?;

    // 配置网格和坐标轴
    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    chart
        .draw_series(LineSeries::new(
            (1..=30)
                .zip(temperature.iter::<f64>()?.collect::<Vec<f64>>().into_iter())
                .map(|(x, y)| (x as f64, y)),
            &BLUE,
        ))?
        .label("Temperature")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            (1..=30)
                .zip(exp_weight_avg.into_iter())
                .map(|(x, y)| (x as f64, y)),
            &RED,
        ))?
        .label(&format!("Beta {beta}"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
