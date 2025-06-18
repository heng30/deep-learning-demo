use tch::Tensor;

fn main() {
    let t = Tensor::from_slice2(&[[1, 2], [3, 4]]);
    let t = t * 2;
    t.print();
}
