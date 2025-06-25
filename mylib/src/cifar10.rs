use anyhow::Result;
use flate2::read::GzDecoder;
use reqwest::header::{HeaderMap, ACCEPT, CACHE_CONTROL, USER_AGENT};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use tch::Tensor;
use tokio_stream::StreamExt;

const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const CIFAR10_DIR: &str = "data/cifar-10";
const IMAGE_SIZE: i64 = 32; // 图片大小
const CHANNELS: i64 = 3; // 每张图片的通道数

#[derive(Debug)]
pub struct Cifar10Dataset {
    pub dir: String,
    pub url: String,

    pub train_images: Tensor,
    pub train_labels: Tensor,
    pub test_images: Tensor,
    pub test_labels: Tensor,
}

impl Cifar10Dataset {
    pub async fn new(data_url: Option<&str>, save_dir: Option<&str>) -> Result<Cifar10Dataset> {
        let url = data_url.unwrap_or(CIFAR10_URL);
        let dir = save_dir.unwrap_or(CIFAR10_DIR);

        let cifar10_dir = Path::new(dir);
        if !cifar10_dir.exists() {
            fs::create_dir_all(cifar10_dir)?;
        }

        let archive_path = cifar10_dir.join("cifar-10-binary.tar.gz");
        if !archive_path.exists() {
            Self::download_file(CIFAR10_URL, &archive_path).await?;
        }

        let train_files = [
            cifar10_dir.join("cifar-10-batches-bin/data_batch_1.bin"),
            cifar10_dir.join("cifar-10-batches-bin/data_batch_2.bin"),
            cifar10_dir.join("cifar-10-batches-bin/data_batch_3.bin"),
            cifar10_dir.join("cifar-10-batches-bin/data_batch_4.bin"),
            cifar10_dir.join("cifar-10-batches-bin/data_batch_5.bin"),
        ];

        let test_file = cifar10_dir.join("cifar-10-batches-bin/test_batch.bin");

        // Extract if files don't exist
        if !train_files[0].exists() {
            Self::extract_archive(&archive_path, cifar10_dir)?;
        }

        let (train_images, train_labels) = Self::read_batches(&train_files)?;
        let (test_images, test_labels) = Self::read_batch(&test_file)?;

        Ok(Cifar10Dataset {
            url: url.to_string(),
            dir: dir.to_string(),

            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    async fn download_file(url: &str, path: &Path) -> Result<()> {
        println!("Downloading {} to {}", url, path.display());

        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36".parse().unwrap());
        headers.insert(ACCEPT, "*/*".parse().unwrap());
        headers.insert(CACHE_CONTROL, "no-cache".parse().unwrap());

        let client = reqwest::Client::new();
        let mut response_stream = client
            .get(url)
            .headers(headers)
            .send()
            .await?
            .bytes_stream();

        let mut total_bytes = 0;
        let mut file = File::create(path)?;

        while let Some(chunk_result) = response_stream.next().await {
            let chunk = chunk_result?;

            total_bytes += chunk.len();
            println!("Downloaded {total_bytes} bytes");

            file.write_all(&chunk)?;
        }
        Ok(())
    }

    fn extract_archive(archive_path: &Path, dest_dir: &Path) -> Result<()> {
        println!("Extracting {}", archive_path.display());
        let file = File::open(archive_path)?;
        let mut archive = tar::Archive::new(GzDecoder::new(file));
        archive.unpack(dest_dir)?;
        Ok(())
    }

    fn read_batches(batch_files: &[impl AsRef<Path>]) -> Result<(Tensor, Tensor)> {
        let mut all_images = Vec::new();
        let mut all_labels = Vec::new();

        for batch_file in batch_files {
            let (images, labels) = Self::read_batch(batch_file)?;
            all_images.push(images);
            all_labels.push(labels);
        }

        Ok((Tensor::cat(&all_images, 0), Tensor::cat(&all_labels, 0)))
    }

    fn read_batch(batch_file: impl AsRef<Path>) -> Result<(Tensor, Tensor)> {
        let file = File::open(batch_file.as_ref())?;
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;

        // 1个字节的label标签
        let num_images =
            buffer.len() / (1 + CHANNELS as usize * IMAGE_SIZE as usize * IMAGE_SIZE as usize);

        // println!("Numer of images: {num_images}");

        let mut images = Vec::with_capacity(num_images);
        let mut labels = Vec::with_capacity(num_images);

        for i in 0..num_images {
            let start = i * (1 + CHANNELS as usize * IMAGE_SIZE as usize * IMAGE_SIZE as usize);
            labels.push(buffer[start] as i64);

            let image_data = &buffer[start + 1
                ..start + 1 + CHANNELS as usize * IMAGE_SIZE as usize * IMAGE_SIZE as usize];
            let mut image =
                Vec::with_capacity(CHANNELS as usize * IMAGE_SIZE as usize * IMAGE_SIZE as usize);

            // CIFAR-10 stores images in CHW format (channels, height, width)
            for c in 0..CHANNELS as usize {
                for h in 0..IMAGE_SIZE as usize {
                    for w in 0..IMAGE_SIZE as usize {
                        let idx = c * IMAGE_SIZE as usize * IMAGE_SIZE as usize
                            + h * IMAGE_SIZE as usize
                            + w;
                        image.push(image_data[idx] as f32 / 255.0); // Normalize to [0, 1]
                    }
                }
            }

            images.extend(image);
        }

        let images_tensor = Tensor::from_slice(&images).reshape(&[
            num_images as i64,
            CHANNELS,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ]);
        let labels_tensor = Tensor::from_slice(&labels);

        Ok((images_tensor, labels_tensor))
    }
}
