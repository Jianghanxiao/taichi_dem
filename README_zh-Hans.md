# 使用 Taichi DEM 进行工程定量离散单元法仿真
从工程角度使用太极语言完整实现离散元法。

![](Demos/carom/carom.gif)

![](Demos/cube_911_particles_impact/cube_911_particles_impact.gif)

![](Demos/cube_18112_particles_impact/cube_18112_particles_impact.gif)

>使用BIMBase二次开发可视化。BIMBase是北京构力科技研发的用于BIM的图形平台。https://app.pkpm.cn/pbims

## 作者

Denver Pilphis (Di Peng) - 离散单元法理论与实现

MuGdxy (Xinyu Lu) - 性能优化

## 简介

本例提供一个完整的离散单元法(DEM)仿真的实现，考虑了复杂的DEM力学机制，仿真结果达到工程定量精度。

本例使用Taichi语言，加上适当的数据结构和算法，以保障计算效率。

## 新增功能

相比Taichi DEM原始版本，本例增加了如下功能：

1. 二维离散单元法→三维离散单元法；

2. 完整考虑并实现颗粒方位和旋转，预留了建模和仿真非球形颗粒的空间；

3. 实现了墙（离散元中的边界）单元，并实现了颗粒-墙接触解算；

4. 实现了复杂离散元接触模型，包括一个胶结模型（爱丁堡胶结颗粒模型，EBPM）和一个颗粒接触模型（Hertz-Mindlin接触模型）；

5. 因胶结模型已实现，可以使用胶结团块模拟非球形颗粒；

6. 因胶结模型已实现，可以仿真颗粒破碎过程。

## 示例

### 开仑台球

本例展示了开仑台球开局第一球。白球正对其他球运动并发生碰撞，然后球散开。虽然该过程中有能量损失，但是由于没有抗转动力学响应耗散转动动能，所有的球在进入纯滚动状态后将会一直滚动下去。本例可用于验证Hertz-Mindlin模型。

![](Demos/carom/carom.gif)

### 911个颗粒胶结组成的立方体团块撞击平整表面

本例展示了一个立方体胶结团块撞击平整表面的过程。
撞击过程中，团块内部的胶结受力将发生破坏，然后团块碎成碎片，飞向周围空间。
本例可用于验证EBPM模型。

![](Demos/cube_911_particles_impact/cube_911_particles_impact.gif)

### 18112个颗粒胶结组成的立方体团块撞击平整表面

本例与上例相似，唯独团块所含颗粒数不同。本例可用于大规模体系仿真的性能测试。

![](Demos/cube_18112_particles_impact/cube_18112_particles_impact.gif)

## 致谢

感谢谢菲尔德大学化学与生物工程学院Xizhong Chen博士对本研究给予的帮助和支持。
