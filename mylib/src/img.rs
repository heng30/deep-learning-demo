use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Pixel, RgbaImage};

pub struct ImageView {
    pub width: u32,
    pub height: u32,
    pub img: DynamicImage,
}

pub enum ImageChannelType {
    RED,
    GREEN,
    BLUE,
    ALPHA,
}

impl ImageView {
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<ImageView> {
        let img = image::open(path)?;
        let (width, height) = img.dimensions();

        Ok(ImageView { width, height, img })
    }

    pub fn channel<T>(&self, chan_type: ImageChannelType) -> Vec<T>
    where
        T: From<u8>,
    {
        let mut channel = Vec::new();

        // 获取各个通道
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.img.get_pixel(x, y).to_rgba();

                let p = match chan_type {
                    ImageChannelType::RED => pixel[0].into(),
                    ImageChannelType::GREEN => pixel[1].into(),
                    ImageChannelType::BLUE => pixel[2].into(),
                    ImageChannelType::ALPHA => pixel[3].into(),
                };

                channel.push(p)
            }
        }

        channel
    }

    pub fn channel_to_image<T>(chan: &[T], width: u32, height: u32) -> DynamicImage
    where
        T: Into<u8> + Copy,
    {
        assert!(chan.len() as u32 >= width * height);

        let mut channel_img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let pos = y * height + x;
                let p = chan[pos as usize].into();
                channel_img.put_pixel(x, y, Luma([p]));
            }
        }

        image::DynamicImage::ImageLuma8(channel_img)
    }

    pub fn all_channels<T>(&self) -> Vec<Vec<T>>
    where
        T: From<u8>,
    {
        let mut imgs = vec![];
        imgs.push(self.channel::<T>(ImageChannelType::RED));
        imgs.push(self.channel::<T>(ImageChannelType::GREEN));
        imgs.push(self.channel::<T>(ImageChannelType::BLUE));
        imgs.push(self.channel::<T>(ImageChannelType::ALPHA));

        imgs
    }

    pub fn convert_channels_elem_type<T, U>(chans: &Vec<Vec<T>>) -> Vec<Vec<U>>
    where
        T: Copy,
        U: From<T> + Copy,
    {
        chans
            .iter()
            .map(|chan| chan.iter().map(|e| e.clone().into()).collect::<Vec<U>>())
            .collect::<Vec<_>>()
    }

    pub fn channel_image<T>(&self, chan_type: ImageChannelType) -> DynamicImage
    where
        T: From<u8>,
    {
        let mut channel_img = GrayImage::new(self.width, self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.img.get_pixel(x, y);

                let p = match chan_type {
                    ImageChannelType::RED => pixel[0].into(),
                    ImageChannelType::GREEN => pixel[1].into(),
                    ImageChannelType::BLUE => pixel[2].into(),
                    ImageChannelType::ALPHA => pixel[3].into(),
                };

                channel_img.put_pixel(x, y, Luma([p]));
            }
        }

        image::DynamicImage::ImageLuma8(channel_img)
    }

    pub fn all_channel_images<T>(&self) -> Vec<DynamicImage>
    where
        T: From<u8>,
    {
        let mut imgs = vec![];
        imgs.push(self.channel_image::<T>(ImageChannelType::RED));
        imgs.push(self.channel_image::<T>(ImageChannelType::GREEN));
        imgs.push(self.channel_image::<T>(ImageChannelType::BLUE));
        imgs.push(self.channel_image::<T>(ImageChannelType::ALPHA));

        imgs
    }

    pub fn horizontal_gray_images(imgs: Vec<DynamicImage>) -> DynamicImage {
        let (mut width, mut height) = (0, 0);

        for img in &imgs {
            let (w, h) = img.dimensions();
            width += w;

            if h > height {
                height = h;
            }
        }

        let mut start_x = 0_i64;
        let mut combined = GrayImage::new(width, height);

        for img in imgs {
            let img = img.into_luma8();
            let (w, _) = img.dimensions();

            image::imageops::replace(&mut combined, &img, start_x, 0);
            start_x = start_x + w as i64;
        }

        image::DynamicImage::ImageLuma8(combined)
    }

    pub fn vertical_gray_images(imgs: Vec<DynamicImage>) -> DynamicImage {
        let (mut width, mut height) = (0, 0);

        for img in &imgs {
            let (w, h) = img.dimensions();
            height += h;

            if w > width {
                width = w;
            }
        }

        let mut start_y = 0_i64;
        let mut combined = GrayImage::new(width, height);

        for img in imgs {
            let img = img.into_luma8();
            let (_, h) = img.dimensions();

            image::imageops::replace(&mut combined, &img, 0, start_y);
            start_y = start_y + h as i64;
        }

        image::DynamicImage::ImageLuma8(combined)
    }

    pub fn horizontal_images(imgs: Vec<DynamicImage>) -> DynamicImage {
        let (mut width, mut height) = (0, 0);

        for img in &imgs {
            let (w, h) = img.dimensions();
            width += w;

            if h > height {
                height = h;
            }
        }

        let mut start_x = 0_i64;
        let mut combined = RgbaImage::new(width, height);

        for img in imgs {
            let (w, _) = img.dimensions();

            image::imageops::overlay(&mut combined, &img, start_x, 0);

            start_x = start_x + w as i64;
        }

        image::DynamicImage::ImageRgba8(combined)
    }
}
