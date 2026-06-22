# 使用介绍
本示例展示了控制头部、腰部、手臂和底盘的基本方法。通过调用python SDK中的API，实现控制头部、腰部、手臂和底盘的控制。<br/>
注意：此种控制方式不接运控，如需运控接口，请参考motion_control
# 使用方式
在终端中执行以下命令：
```bash
robot-controller
```
即可启动控制程序:
输入he或head
以下为输出示例，单位为角度与角度：
```bash
Enter the part and params: he
Current head state: [0.0, 25.000457337210047], usage: he <angle>,<angle>
```
查询腰部俯仰和升降状态信息
输入wa或waist
以下为输出示例，单位为角度与厘米：
```bash
Enter the part and params: wa
Current waist state: [0.0, 0.0], usage: wa <angle>,<height>
```
头部和腰部控制示例
```bash
Enter the part and params: wa 25,30
Current waist state: [0.5230003543799336, 0.3], move to [0.4363323129985824, 30.0]
Enter the part and params: he 10,10
Current head state: [0.0, 25.00030474932202], move to [0.17453292519943295, 0.17453292519943295]
```
夹爪使用示例
输入gr或gripper
以下为输出事例，单位为左手和右手的张合程度：
```bash
Enter the part and params: gr
Current gripper state: [35.52, 35.36], usage: gr num,num
```
夹爪控制示例
```bash
Enter the part and params: gr 50,50
Current gripper state: [35.52, 35.36], move to [50.0, 50.0]
```
手臂控制示例
输入ar或arm
以下为输出事例，单位为角度与角度：
```bash
Enter the part and params: gr 0.8,0
Current gripper state: [35.480000000000004, 35.36], move to [0.8, 0.0]
```
灵巧手使用示例<br/>
输入hand或ha，此时需要输入灵巧手所有关节的角度值hand_as_gripper或ha_as_gr，此时只需要输入0-1,0-1的模式，即可按照控制夹爪的方式来控制灵巧手

输入re或者reset即可复位
