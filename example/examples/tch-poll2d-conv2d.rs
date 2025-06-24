use anyhow::Result;
use mylib::img::ImageView;
use tch::{
    Device, IndexOp, Tensor,
    kind::Element,
    nn::{self, Module},
};

const INPUT_IMG: &str = "data/lena.png";
const OUTPUT_IMG: &str = "target/output.png";

fn main() -> Result<()> {
    // 设置随机种子以便结果可重现
    tch::manual_seed(5);

    let img = ImageView::new(INPUT_IMG)?;
    let chan_imgs = img.all_channels::<u8>();

    // 创建神经网络
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut config = nn::ConvConfig::default();
    config.stride = 1;
    config.padding = 1;

    // 输入4个通道，输出3张特征图
    let conv = nn::conv2d(&vs.root(), 4, 3, 3, config);

    let chan_imgs = ImageView::convert_channels_elem_type::<u8, f32>(&chan_imgs);

    let input = channel_images_to_conv2d_tensor(
        chan_imgs
            .iter()
            .map(|item| item.as_slice())
            .collect::<Vec<_>>()
            .as_slice(),
        img.width,
        img.height,
    );

    // 在tensor前添加一个维度
    let input = input.unsqueeze(0);

    // 进行卷积
    // 输入格式(batch, channel, height, width)
    let output = conv.forward(&input);

    // 最大池化
    let output_max_poll = output.max_pool2d([2, 2], [1, 1], [0, 0], [1, 1], false);

    // 平均池化
    let output_avg_poll = output.avg_pool2d([2, 2], [1, 1], [1, 1], false, true, None);

    let mut imgs = vec![];
    for output in [output, output_max_poll, output_avg_poll] {
        // 移除第一个维度
        let output = output.squeeze();

        assert_eq!(3, output.size()[0]);
        println!("{:?}", output.size());

        let mut output_imgs = vec![];
        for i in 0..3 {
            let chan_img = output.i(i);
            let (c, w, h) = conv2d_tensor_to_channel_image(&chan_img)?;
            output_imgs.push(ImageView::channel_to_image(&c, w, h));
        }

        imgs.push(ImageView::horizontal_gray_images(output_imgs));
    }

    let output_img = ImageView::vertical_gray_images(imgs);
    output_img.save(OUTPUT_IMG)?;

    Ok(())
}

fn channel_images_to_conv2d_tensor<T>(chans: &[&[T]], width: u32, height: u32) -> Tensor
where
    T: From<u8> + Copy + Element,
{
    let dim = chans.len() as i64;

    // 转换为(C, H, W)
    let img_tensor = Tensor::from_slice2(&chans).reshape([dim, height as i64, width as i64]);

    img_tensor
}

fn conv2d_tensor_to_channel_image(t: &Tensor) -> Result<(Vec<u8>, u32, u32)> {
    assert_eq!(2, t.dim());
    let size = t.size();
    let height = size[0] as u32;
    let width = size[1] as u32;

    let mut data = vec![];
    let t = t.reshape([1, -1]).squeeze();

    for v in t.iter::<f64>()? {
        data.push(v.abs().min(255.0).max(0.0).round() as u8);
    }

    Ok((data, width, height))
}
