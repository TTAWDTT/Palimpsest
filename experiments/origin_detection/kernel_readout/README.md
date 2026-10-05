# Kernel readout — fourteenth development protocol

目的：检验现有像素表示是否有线性/单Gaussian头无法表达的类别边界，不假定核映射产生物理不变性。
固定组合499维fit加权标准化，γ=1/499，seed20261006，256个iid Gaussian频率，cos/sin共512维。
没有抽样landmark、真假引导频率或selection带宽选择。推理不看目标批次。
[Rahimi/Recht NIPS2007全文](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf)正文与附录已读；
Bochner核近似依据不是处理后准确度定理，512维也未经整个输入域的误差认证。

四候选：Fourier512／原统计+Fourier1011 × ERM／组风险+配对。
第十三轮相同ridge0.01、pair0.1、temperature0.1、同一48组、1260fit/420threshold/420selection。
头仍用fit特征尺度标准化，故其正则化不是未经改动的Gaussian RKHS范数。
来源标签置乱在组合+both头上作阴性控制。
map先逐图片计算，再计算同源mapped差分；不能将非线性map直接作用于原差分。
评测分数必须由完整portable KernelRule直接从原499维重算，与缓存mapped头分数完全一致。

先常模、特征平移kernel内积、指定点Gaussian核近似、同均值/方差的非线性人工样例、序列化检查。
然后完整fit映射加ERM成本小试；不读取原图像、不解封RR reserved，不修改旧缓存/代码/收据。

```powershell
.venv/Scripts/python.exe -m experiments.origin_detection.kernel_readout.run_iteration --pilot
.venv/Scripts/python.exe -m experiments.origin_detection.kernel_readout.run_iteration
```

结果在`work/robust_statistics/kernel_readout`，拒绝覆盖；准确度失败不投入最终速度基准或盲集调参。
