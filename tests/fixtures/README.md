# 回归样本

样本用于重现已发生的道路、控制和任务问题。测试应读取这里的配套标定，不依赖私有 `.maixpy/` 目录或当前设备姿态。原始完整记录仍保留在本机 `.maixpy/`，这里仅保存测试实际读取的输入。

- `camera_pose_20261031_191327.json`：2026-10-31 19:13固定姿态的算法回归标定，原样冻结自2026-11-04现场重标定前的配置。历史轨迹、固定几何、边界预筛及原生等价测试使用它，避免随现场相机姿态变化而改变回归输入。部分孤立历史图片仅用于该固定投影下的数值等价诊断，此标定不代表这些图片的拍摄真值；已有配套拍摄标定的样本继续使用各自标定。应用加载与部署配置测试仍读取当前生产标定。
- `synthetic_road_calibration.json`：早期30/60cm投影，用于生成一致的合成单边/双边道路，不依赖已移除的Duckietown实现。
- `curve_recordings_20261021/`：静止单侧边线的三个原始帧，配套上述合成标定。
- `curve_recordings_20261023/`：固定姿态、反光、明暗回退和弱边界片段；`calibration.json`来自当天固定姿态标定，5mm采样的专项测试显式设置分辨率。
- `road_model_sequence/`：原memory-root-cause/reproduction-trace的794—878帧，PNG像素原样保留。`telemetry.json`只保留测试需要的captured、v、omega，不携带旧模型输出。配套`road_model_recording_calibration.json`。
- `crosswalk_recordings_20261024/`：横道正样本及单白线/普通道路反例，配套`crosswalk_20261024_calibration.json`。
- `crosswalk_20261030/`：弯道后横道对比度回归。
- `start_signal_jitter_20261031/`：未切灯画面，验证ARM后噪声不触发发车；旧模板匹配样本已删除。
- `lap_line_20261031/`、`lap_marker_20261031/`：计圈横线与普通白线的回归输入。
- 其余`curve_*.json/png/jpg`：对应具名测试的最小几何、命令或感知回归样本；`*_reference.py`为批量/原生加速的独立参考实现，不能作为重复代码删除。
- `right_bend.jpg`仍用于当前边界预筛与原生核等价测试，故保留。

缺失必需样本应导致测试失败；可选C核未构建时，原生等价测试会跳过，构建后应全部执行。

- `bend_far_support_20261101.json`：v0.6.54实车325/329帧的已确认控制路径；近横向道路被FAR_SUPPORT最近点误导为近直行。仅用于选点回归，不代表修复后的实车轨迹。

- `esp_start_signal_history/`：16张真实灯光开关样本，原始320×240面板，ROI标注和来源见manifest.json；仅作为RGB565迁移回测，不能代表ESP32实测准确率。

- `yield_cancel_road_20261107.json`：v0.6.69实际停车日志的10帧最小道路状态；道路已有效观测、横道处于APPROACH，取消搜路却持续截断巡线。测试提供的PWM是协调器输入，不是实测车轮输出；不代表实车轨迹回放。

- `right_road_stall_20261107/`：实车卡住时的原图面板及配套标定，源图为预览JPEG解码，不冒充无损传感器帧。覆盖绿色隔带两侧路面误配边沿，含镜像和不满足分离证据的反例。
# Crosswalk partial-view regression (2026-11-11)

`crosswalk_20261111/three_bars.png` is the raw panel from the paused live preview where four-bar acquisition no longer found a crossing. Its capture-time project calibration and an explicitly reconstructed four-bar reference are included. The old app transported the line plane but did not transport its logged corners, so `reference.json` documents the longitudinal-only reconstruction assumption. This fixture verifies image-level reacquisition of three bars against that template; it does not validate the complete driving trajectory, exact previous-frame pose, or physical stopping distance.

- `crosswalk_unverified_20261111.json`：v0.6.94帧4199的真实横道卡死状态；位置区间误差54.2cm、距离估计31.5cm。时间戳归零，用于验证不确定状态首帧交回低速跟线/寻路且不伪报合法完成；不是完整实车轨迹回放。

- `single_edge_stall_20261111/`：v0.6.97用户STOP后抓取的原图、对应标定及同轮帧1403最后有效中线。两份几何/图像不是同一帧；用于单边推断和历史淘汰门槛的独立回归，不冒充完整位姿回放。

- `persistent_bend_exit_20261111.json`：v0.6.98帧2033的真实反证、主路线和当前候选，另含前一轮停车后远处横向道路快照。用于防止明确绿色时继续直行及远路错误替换；不同快照不构成完整位姿回放。

- `body_clearance_speed_20261111.json`：v0.6.99现场帧1849与2027的独立控制快照；原规划参考速度超过已有车身预测限速，覆盖两种转向。边界风险为算法预测，非逐帧人工压线标注，也不是连续位姿回放。
