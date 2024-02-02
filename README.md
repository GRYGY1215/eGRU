# eGRU
Code of paper [A Novel Extreme Adaptive GRU for Multivariate Time Series Forecasting]

The Dozerformer achieve SOTA on nine benchmarks.

## Introduction
Multivariate time series forecasting is a critical problem in many real-world scenarios. Recent advances in deep learning have significantly enhanced the ability to tackle such problems. However, a primary challenge in time series forecasting comes from the imbalanced time series data that include extreme events.
Despite being a small fraction of the data instances, extreme events can have a negative impact on forecasting as they deviate from the majority. However, many recent time series forecasting methods neglect this issue, leading to suboptimal performance.
To address these challenges, we introduce a novel model, the Extreme Event Adaptive Gated Recurrent Unit (eGRU), tailored explicitly for forecasting tasks. The eGRU is designed to effectively learn both normal and extreme event patterns within time series data.
Furthermore, we introduce a time series data segmentation technique that divides the input sequence into segments, each comprising multiple time steps. This segmentation empowers the eGRU to capture data patterns at different time step resolutions while simultaneously reducing the overall input length.
We conducted comprehensive experiments on four real-world benchmark datasets to evaluate the eGRU's performance. Our results showcase its superiority over vanilla RNNs, LSTMs, GRUs, and other state-of-the-art RNN variants in multivariate time series forecasting.
Additionally, we conducted ablation studies to demonstrate the consistently superior performance of eGRU in generating accurate forecasts while incorporating a diverse range of labeling results.

## Usage
The eGRU cell and eGRU layer are implemented in PyTorch at ./models/eGRU/eGRU.py. Examples for eGRU cell and layer are presented as follows:
   ```
   # eGRU cell Example
   input = torch.randn(6, 3, 10)
   input[:3, :, 9] = 1
   input[3:, :, 9] = 0
   hx = torch.randn(6, 20)
   outputs = []
   rnn = eGRU_cell(10, 20)
   for i in range(6):
       print(i)
       output, hx = rnn(input[i], hx)
       outputs.append(output)
   ```
   ```
   # eGRU layer Example
   input = torch.randn(5, 3, 11)
   input[:, :2, 10] = 1
   input[:, 2:, 10] = 0
   rnn = eGRU(10, 20, 2)
   h0 = torch.randn(2, 6, 20)
   output, hidden_states, layers_out_nor, layers_out_ext = rnn(input)
   ```

