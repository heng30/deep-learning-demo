use approx::assert_abs_diff_eq;
use ndarray::{Array, ArrayD, array};

/// Sigmoid 函数
pub fn sigmoid(x: &ArrayD<f32>) -> ArrayD<f32> {
    1.0 / (1.0 + (-x).mapv(f32::exp))
}

/// Sigmoid 函数的反向传播
///
/// 参数:
/// - grad_output: 从上一层传来的梯度
/// - sigmoid_output: 前向传播时 sigmoid 的输出值
///
/// 返回: 本层的梯度
pub fn sigmoid_backward(grad_output: &ArrayD<f32>, sigmoid_output: &ArrayD<f32>) -> ArrayD<f32> {
    // sigmoid 的导数为: σ'(x) = σ(x) * (1 - σ(x))
    // 复合函数的链式求导。所以反向传播时梯度为:
    //     grad_output * σ'(x) = grad_output * sigmoid_output * (1 - sigmoid_output)
    grad_output * sigmoid_output * &(1.0 - sigmoid_output)
}

fn main() {
    // 测试前向传播
    let x = array![[0.0, 1.0], [-1.0, 2.0]].into_dyn();
    let output = sigmoid(&x);

    // 已知 sigmoid(0)=0.5, sigmoid(1)≈0.731, sigmoid(-1)≈0.269, sigmoid(2)≈0.881
    assert_abs_diff_eq!(output[[0, 0]], 0.5, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], 0.7310586, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], 0.26894143, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 1]], 0.880797, epsilon = 1e-4);

    // 测试反向传播
    let grad_output = Array::ones(x.raw_dim());
    let grad = sigmoid_backward(&grad_output, &output);

    // 验证梯度计算是否正确
    // 对于每个元素，梯度应该是 output * (1 - output)
    for (&o, &g) in output.iter().zip(grad.iter()) {
        assert_abs_diff_eq!(g, o * (1.0 - o), epsilon = 1e-4);
    }

    // 测试非单位梯度的情况
    let grad_output = array![[2.0, 3.0], [0.5, 1.5]].into_dyn();
    let grad = sigmoid_backward(&grad_output, &output);

    // 验证梯度计算是否正确
    assert_abs_diff_eq!(grad[[0, 0]], 2.0 * 0.5 * 0.5, epsilon = 1e-4);
    assert_abs_diff_eq!(
        grad[[0, 1]],
        3.0 * 0.7310586 * (1.0 - 0.7310586),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        grad[[1, 0]],
        0.5 * 0.26894143 * (1.0 - 0.26894143),
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(
        grad[[1, 1]],
        1.5 * 0.880797 * (1.0 - 0.880797),
        epsilon = 1e-4
    );
}
