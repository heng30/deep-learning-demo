use tch::Tensor;

fn main() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();

    println!("\n===========\n");

    let t = Tensor::from_slice2(&[[1, 2], [3, 4]]);
    let t = t * 2;
    t.print();
}
