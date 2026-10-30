曲线巡线 v0.6.24：连续中线与动态前瞻 Pure Pursuit

2026-10-30集成：车身中心标定、转向增益0.85、弯道基础120/直道最高200、
可信历史中线衔接、横道局部对比度检测。保持手动START，综合实车复验待完成。

运行与部署
  main.py 为入口；build.py --drive 生成 standalone.py 和安装器。
  安装器保存旧版本并原子替换。部署、重启只进入预览，需手动START。
  网页 http://设备IP:8000 提供原图、巡线叠加、地面投影和START/STOP。

统一处理链
  原图 -> 白边候选 -> 单边法线偏移/兼容双边融合 -> LocalRoadModel
  -> 已见曲线的运动预测 -> 最近道路参考 -> 动态前瞻点 -> Pure Pursuit -> 左右轮指令。
  不区分某个弯道、圈数、路段或来源侧来切换控制算法。

拟合
  geometry.py 用弧长参数的局部二次模型拟合 x(s)、y(s)，可表达S形道路。
  已知路宽30cm，单边沿法线偏移半个路宽；两边重叠且兼容时融合。
  每帧复用地面投影和区域掩膜，依次尝试局部对比度及原始亮度视图。
  候选须有路外绿色与路内证据；整条轮廓不可能有绿色探针时提前略过。
  短片段统一按RoadPolicy.minimum_segment_m判定，不再另设10cm门槛
  或按前向切线正负丢弃弯道片段。拟合器仍拒绝退化、尖点和自反转曲线。
  多个候选使用几何相容性、近端支持与历史关联选取，冲突须重新确认。

道路历史与预测
  road_model.py保存每个道路采样点的真实观测时间、质量、误差和未见行程。
  ground_motion.py用双向光流、地面投影与SE(2)补偿此前已拟合的曲线。
  启动时只查询当前帧可信连续段，行驶时允许有界历史支持；两者不删除历史。
  暂时看不到边线时，可沿仍可信的历史曲线短时继续。预测点不会变成新
  观测，seen时间不刷新，误差及时间/未见行程界限保持有效。
  已删除从远处单段向近处做多项式盲延长的_control_prefix：两个拟合窗口
  数值一致，不能证明未见弯道几何正确；用户图像已反证旧假设。
  不再生成任何模型支持之外的控制曲线点。没有历史近段时，追踪当前
  已拟合曲线上选前瞻点，不能凭空补出近处转弯。
  绿色=本帧真正更新的中线；橙色虚线=此前看见、当前由运动补偿得到的
  历史中线。橙色不是前方未知道路的猜测。path_observed_mask逐点标注。

动态前瞻控制
  最近点只定位连续道路的起始参考，不直接作为Pure Pursuit目标。
  前瞻半径=max(整车长度, 2*速度上界*(电机响应时间+当前处理时间+视觉采样间隔))，
  并受最大目标距离限制。2是保守的响应裕量，不是声称已完成底盘系统辨识。
  整车长度18cm由用户测量；电机响应时间0.4s沿用现有工程配置。
  速度上界取当前有效地面光流速度加两倍位置误差/采样间隔。
  光流暂缺时仅保留上次速度上界用于前瞻长度，不冒充实测运动来变换道路。
  在最近参考之后，选择与实际前瞻圆第一次相交的有支持中线点。
  前瞻圆超出连续支持时取末端并降速；整个支持在圆外时取最近参考并降速。
  不跨过曲线退出前向域的位置去寻找后面的另一分支，也不生成盲延长线。
  该方法借鉴Nav2的速度自适应Pure Pursuit；没有引入ROS运行时。
  参考：https://github.com/ros-navigation/navigation2/tree/main/nav2_regulated_pure_pursuit_controller
  实测接收至计算延迟仍作SE(2)补偿，然后重新选前瞻点。
  红十字显示实际控制目标；steering中记录前瞻半径、目标模式及命令时坐标。
  target.nearest_reference_m单独保存最近参考，便于与实际追踪目标区分。
  车身未来包络使用相同前瞻规则，避免安全预测与真实控制使用两套目标。
  Pure Pursuit曲率k=2y/(x²+y²)，经验放大系数保持1.0。
  左右轮加速与峰值增力都共同缩放，保持指令比例；PWM不是实测轮速。
  这次摆动日志中近点距离从约28cm降到7cm，导致旧控制曲率达到-16/m；
  新控制不再随近端片段出现而把前瞻半径缩到6cm。
  闭环仿真包含采样、计算延迟和一阶电机滞后，不能代替真实底盘验证。

配置与几何
  runtime.control_frame：底盘参考点在镜头后方5.5cm，轴距11cm。
  车体前/后/半宽相对参考点为7.5/10.5/8.5cm，依据用户测量。
  横道沿原镜头地面坐标计算，镜头距车头2cm、车尾16cm。
  curve_analysis_window_m=0.30只用于曲率分析与努力降速，不限定目标距离。
  observation_search_near_m=0.40用于亮度视图搜索的近端覆盖判定。
  参数集中于RoadPolicy/runtime/calibration；单位为米、秒、弧度。
  工程误差界限未做概率标定，光流也不是轮编码器。

保护与横道
  道路模型负责一次融合/冲突确认；已删除PursuitReference重复确认层。
  独立MotorChannel与UNO保留命令租约、通信失效归零和手动STOP。
  当前视觉处理预算280ms，未为让车运行而放宽。接收时间不等于曝光时间。
  车身包络只在真实相机覆盖且有道路支持处判断，不把推算段当成观测。
  横道任务独立：线前停稳、红灯、3.3秒礼让后自动继续。异常锁存仍停车。
  疑似横道第一次出现就应用相同刹车距离/速度上限，三帧确认只授权礼让事件，
  不再在确认阶段继续全速靠近。停车确认时限和规则距离窗口保持不变。
  已验证的停车位置证据与横道观测跟踪分离：位置区间随实测地面刚体运动和
  误差传播，间隔内的平移与旋转漂移也计入。证据创建仍需原有定位质量门槛。
  连续实测静止、零轮速回执、红灯和合规距离区间共同支持礼让计时；未确认
  初始位置时不能靠等够时间放行。倒退、移动或证据丢失重置停车计时。
  /vision.crosswalk新增stop_verified、stop_clearance_interval_m、stop_blockers。
  每个横道独立清零计时；锁存故障后不再继续改写故障现场的距离证据。
  已通过183项curve和37项crosswalk测试。带延迟仿真旧版在横道后0.4cm停，
  新版在横道前4.4cm停并完成礼让；这是仿真，不代表实车测量。
  v0.6.6已部署并验证停车状态零输出；完整横道实测尚待从横道前重新进行。
  证据及源码备份：.maixpy/crosswalk-stop-fix/。

模块
  tracker.py 观测、道路模型与选点适配；geometry.py纯几何和预测；
  road_model.py持续道路/边界身份；ground_motion.py运动；
  steering_control.py运动时间对齐、偏航反馈与车身包络；
  controller.py几何轮速、共同爬升和运行状态；crosswalk.py横道任务；
  motor_channel.py/drive_link.py电机通信；preview_transport.py网页。

审查与验证
  删除固定前瞻圆、曲率选前瞻距离、正常/回退双模式、目标弧长限速、
  独立目标确认器、重复运动变换函数、未使用常量与10cm/朝向特判。
  应用源码净减少约150行（不含打包文件）。审查依据与快照：
  .maixpy/nearest-refactor/。190项回归通过；24个板上固定图像样本
  中位143.4ms、P95 187.9ms、最大269.6ms。实际动态负载仍需监测。
  历史录像为开环传感器回放，不代表采用新控制后的实际路径。
  最近点可能与旧远前瞻目标方向不同；新的实际弯道效果尚待实车验证。

v0.6.1 实际错误图回归
  tests/fixtures/curve_unfounded_prediction.png：用户指出橙线沿远处直段伸到
  路外。现在冷启动只保留真实拟合曲线，不再画出/追踪虚构近段；遮挡测试
  验证有效光流仍能平移旋转历史曲线，且观测时间不变。162项curve及28项
  crosswalk通过。新控制输入及橙线定义需要重新实车验证，尚未证明全程。
  控制遥测新增control_target_source=OBSERVED/HISTORY PREDICTION。

性能与持续诊断（v0.6.3）
  本次用正常、弯道、横道及超时位置的18张录像原图做MaixCAM2同机交替回放。
  识别候选数量、道路模式、置信度、中线逐帧等价；没有缩图、裁剪视野、
  减少亮度尝试、放宽阈值或改变Pure Pursuit/停车预算。
  轮廓法线探针、循环轮廓拆分与朝路面检查合并到curve_boundary_chains。
  前瞻/最近点搜索合并到curve_select_pursuit。两者均保留独立Python参考回退。
  曲线二次拟合、历史观测时效、横道任务与电机租约的语义保持不变。
  相机地面覆盖最近距离由固定标定一次计算，避免每帧重建同一数组。
  关闭cProfile后的板端同图基准：道路提取中位数66.6->28.6ms，
  最终同图控制链和逐阶段数据见.maixpy/speed-profile/v064-summary.json。
  回放控制链不包含取帧和显示，不能等同于实车整帧延迟或动态行驶效果。
  cProfile增加执行开销，只用于找热点，不用于上述最终耗时对比。
  /vision.profile_ms 与 /status.readiness.profile_ms提供每帧耗时：
    camera_read、image_conversion、motion、perception、model、target、
    crosswalk、steering、control_pipeline、supervision_motor_telemetry、
    overlay、display_conversion、display_show、stream_encode、logging、
    frame_processing和frame_period。
  control_pipeline包含运动/识别/模型/目标/横道/转向，不能再次与这些子项相加；
  frame_processing包含控制链和输出，frame_period是相邻图像接收间隔。
  target.photometric_passes记录本帧实际尝试的亮度视图数量，便于区分慢帧。
  固定摄像头实测与回放明细保存在项目.maixpy/speed-profile。

  车身包络在保持未来轨迹不变的前提下，将28个角点批量投影与判断覆盖。
  已对120种曲线/可见性/命令时姿态与原先逐姿态实现核对。

  v0.6.4进一步合并历史曲线关联的弧长/切线数值计算；连续、有限、重叠、
  距离和方向判据与Python参考保持一致。C调用包装器在持有连续、定型、已校验
  形状的局部数组期间传递地址，避免对每个参数重复ndpointer检查。
  同帧相同曲线及法线偏移复用纯几何结果，每次返回独立元数据字典，避免
  多亮度视图的score/side/overlap字段互相污染。缓存只在geometry_frame内有效。
  此更改没有让旧观测变新，没有跨帧缓存道路识别。

  最终报告：.maixpy/speed-profile/report.txt。118帧停车采样中控制最大约279ms、
  0帧超280ms；完整周期P95约318ms，仍有输出和停车查询长尾。未执行行驶验证。

中线覆盖切换（v0.6.7）
  正常长重叠、真正几何冲突、重叠不足和融合失败分别记录。
  仅在旧支持有效、新候选质量足够时，检查旧尾/新头是否落在各自末端窗口，
  距离是否在当前观测误差以内、方向是否一致、共享部分是否全部相容。
  不生成接头插值曲线，不跨越明显空缺，也不把反向、交叉或平移道路当成续接。
  有有效地面运动与接头证据时，使用原有confirmation_evidence确认新曲线，
  可在移动中完成；证据不全时仍走原来的停车确认，不全局缩减确认帧数。
  确认期间只允许沿旧支持预测，帧时间/盲行距离/误差界限不变。过渡限速同时
  考虑剩余时间、剩余距离、刹车余量及实测速度，不得以未知运动授权继续。
  启动时仍只允许当前已确认的观测，不使用待确认线或预测段凑启动计数。
  target新增association_kind、handoff_evidence、transition_pending，待确认时
  记录association_prior_path_m及candidate_path_m，以便下一轮直接核对几何。
  已通过196项curve及37项crosswalk测试。85帧真实录像前后同为83次观测、
  2次保守确认、0次道路失效，该录像未触发新的短重叠切换，不能代表实车收益。
  最近两段日志缺少候选曲线原始点，不能精确反事实重放其停顿变化。
  本轮主要优化中线切换；横道测速短暂停顿与前瞻时间变化尚需独立验证。

短观测与停车确认（v0.6.8）
  短段完全落在可信旧曲线内时，以连续投影检查全部采样点的距离、方向和
  实际覆盖弧长，只替换真实观测区间，不丢弃或刷新未见的旧尾段。
  路径冲突时先输出零轮速；收到新鲜零速回执且光流确认静止后，再开始
  有界道路确认。刹停阶段不能消耗停稳确认帧数，也不能无限重启。
  刹停等待界限由已有速度响应时间与视觉预算相加确定；真正不一致的候选
  仍必须重新确认。故障停车后保持锁止，需明确START重新启动。

任务恢复与完赛优先（v0.6.9）
  道路确认次数只作诊断，不再因达到2*N-1次而锁止。道路拟合、历史有效性、
  转向和车身包络仍独立决定能否输出非零速度；监督器不能制造驾驶指令。
  有真实候选或模糊边界时保持任务运行、零速重新识别；中线重新确认且
  命令及时可用后自动继续，无需再次START。原有可信历史低速续行保持有效。
  只有及时画面持续找不到任何边界/候选、同时历史已不可用，才累计原有
  missing_command_s并报ROAD NOT FOUND；次数、历史过期、单个慢帧均不单独
  等同于迷路。慢帧仅允许零速等待；独立电机租约与UNO看门狗保持有效。
  STOP、通信故障、横道任务故障等仍可终止运行，不自动重新解除电机锁止。
  ROAD NOT FOUND仅表示无可信道路，不声称仅凭这个判据识别了终点。

Active recovery (v0.6.10)
  Try other feasible pursuit goals on the same supported curve before search.
  Every candidate retains the body-envelope and observation checks.
  road_recovery.py prioritizes measured corridor coverage. When the camera
  cannot cover the chassis, a recent local pose inference can authorize only
  a small search rotation, with curvature/distance uncertainty accounted for.
  LOCAL_POSE_ESTIMATE is not observed free space and never creates a driving
  path. Its evidence can persist only through a bounded search with continuous
  valid motion and growing uncertainty. Missing transforms revoke the map.
  Search goals are initial heading +/-10 degrees, then centre; each pulse has
  an independent <=80ms motor lease and is followed by zero and measured rest.
  Drift >1.5cm, unsupported space, excessive error or search-duration exhaustion
  prevents further pulses. Actual motor angle response still needs testing.
  Normal motor/UNO watchdogs and explicit STOP remain effective.
  Completed crosswalk dwell no longer requires road-ready at the same instant.
  Temporary speed/position issues no longer terminate the whole mission.
  A physical yield without verified legal position records yield_completed=true
  and yield_verified=false plus issues. It is never a successful legal-stop
  claim. Motion, zero ACK and brake-lamp evidence are still required for dwell.
  Search rotation is disabled while a crossing owns the stop.
  HTTP/JPEG failures affect preview only, retaining local touchscreen control.
  Offline tests and stopped deployment do not constitute physical turn testing.

Soft road-loss recovery (v0.6.11)
  ROAD NOT FOUND retains the running mission at zero output. Only trusted
  consecutive current-road observations and a valid follow command can resume
  normal driving. An approved bounded search pulse can operate within this
  wait; it does not clear the road-loss evidence or rearm any hardware fault.
  Manual STOP, emergency stop and motor/communication faults never auto-rearm.
