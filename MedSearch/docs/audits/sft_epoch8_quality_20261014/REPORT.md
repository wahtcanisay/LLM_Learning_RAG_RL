# 8 epoch 冷启动回答质量抽查

日期：2026-10-14。仅检查已有产物，没有追加训练或修改评分、数据发布与模型指针。

## 判断

建议结束本轮冷启动，保留 CE+Top32 epoch8 作为当前 RL 初始化候选。继续使用同1000条轨迹可能还有小幅拟合收益，现有结果不足以支持稳定的开发集收益，也不能据此断言超过8轮绝无收益。

- Top32 dev verifier判对数：1/3/5/8epoch为85/96/92/100，非单调；纯CE为85/97/91/91。
- Top32第7/8轮训练总loss的日志窗口均值：0.4421107→0.4390943，下降约0.68%；CE项0.4109621→0.4081329。这里是训练日志窗口均值，不是独立验证loss。
- 调度已到尾部：checkpoint1000的计划学习率为0，第8轮末次更新日志约2.50e-10。低学习率也会造成末期变化变小，不能把这一点当作容量耗尽的证明。[HF cosine调度说明](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup)。若以后增加预算，需要明确新的调度，并做新的dev比较。

## 抽查范围

固定seed20261014，按两种评分都通过、都未通过、verifier单独通过、词面单独通过、未被环境接受最终答案五类，抽取20条唯一ID。阅读生成文本，并重点回看下述例子的实际可见检索段落；这是一组诊断样例，不是医学专家审核或随机总体准确率估计。

原始样例保存为sampled_cases.jsonl，完整来源为outputs/distillation/sft_epoch_comparison_20261013/epoch8/rollouts.jsonl。抽样与统计命令在tmp/review_sft_epoch8.py、tmp/show_sft_quality_cases.py和tmp/save_sft_quality_metrics.py。

## 生成质量：好的部分

- 392/396题的最终答案被环境接受（98.99%）；1题包含无效动作。动作接口已比较稳定，但不等于全篇think标签都规范。
- 1227题：问心肌电与机械合胞体的结构基础，模型搜题目关键结构，Doc2明确写intercalated disks及机械/电连接，模型随后总结并提交intercalated disks。答案、题目和可见证据对应清楚。
- 1727题：问两条静脉汇合形成什么，Doc1明确给出external jugular vein，模型提取同一名称作答。属于适合当前系统的直接证据抽取。

## 生成质量：主要不足

1. **答案吻合，解释仍错误。** 2090题的参考答案与输出都为Thiamine，但模型先编出题干未给出的高血糖、胰岛素缺乏，再用维生素缺乏和Wernicke-Korsakoff解释新生儿焦糖味尿。此症状组合应重点考虑MSUD的支链氨基酸代谢机制，不能用上述编造病因支持答案；相关特征与硫胺素反应型见[GeneReviews](https://www.ncbi.nlm.nih.gov/sites/books/NBK1319/)。这是对该试题解释的审查，不是患者诊疗建议。两种答案评分都通过了该例，说明终局答案得分不能验收推理真实性。
2. **被局部词面相似的噪声带走。** 1745题问成瘾者可能使用什么物质，模型最终答furuncular myiasis。可见Doc3描述真实皮肤寄生虫病，与题目中的皮肤爬行感有词面关联，模型却忽略问题要求的答案类型；Doc1还有co-caine的OCR断词。问题包含检索噪声和证据选择失误，不应全归因于训练轮次。
3. **把检索结果拼成答案，缺少筛选。** 3445题问organisms，模型把Doc2的obesity以及Doc3参考文献中的病名一并列为organisms。该旧dev题面的emergency/reemergence表述也有歧义，参考答案集合未必穷尽所有可能例子；但把obesity当作organism的类别错误，从题面和文档本身就能识别。
4. **搜索失败后复读。** 8轮共107题实际检索两次；48题第二次与第一次查询相同（仅规范大小写/空白），且可见观察SHA256相同，未获得新信息。全部396题均至少检索一次；是否应直接作答还取决于固定prompt与任务要求，不能仅据此断言搜索多余。8轮重复执行查询占全部题12.12%，占两次搜索题44.86%；1轮为23题，3轮43题，5轮51题。更长SFT没有显示出持续改善该行为。
5. **预算结束时仍有停止失败。** 1640、39886、19910在两次搜索后收到明确的最终作答提示，仍生成新的search动作，未完成被接受的最终答案。第三个search没有实际执行，环境仍遵守两次上限。

部分生成段落还有多余的think闭标签，以及“证据不相关→最终照样采用”的自相矛盾，表面流畅性高于论证可靠性。

## 评分口径发现

1342题的原始文本含有正确的answer标签，但动作未被环境接受，final_answer为null；verifier仍从原始文本抽取答案并判对。原始评分保留。

2026-10-14进一步定位：1342并非标签未闭合或生成预算截断。真实token解码为`<answer>Escherichia coli</answer>.`，闭标签末尾与句号共享一个`>.` token。`truncate_generated_action`的字符边界文本可被`parse_action`接受，但离线`generate_actions(capture=True)`额外比较完整保留token的解析结果，因尾随句号将该动作置为None；训练loop没有这道额外检查，能接受字符边界文本。因此该例说明离线评测与训练loop的口径差异，不能据此增加RL严格格式门槛。其他3条未回答轨迹均在最后作答机会再次输出search；最终动作分别105/108/129 tokens，均未达到单动作500，整条也未达到response4096。复现命令：`C:\Users\dell\anaconda3\python.exe -m tmp.check_answer_token_boundary_20261014`；原始失败样例查看：`C:\Users\dell\anaconda3\python.exe tmp\inspect_unaccepted_answers_20261014.py`。尚未改解析器、生成预算、评分或核心实现。

增加诊断口径“环境接受最终答案且verifier通过”：Top32 epoch8为99/396，纯CE epoch3为96/396，候选顺序与净差3题均未改变。该口径不是对原评分的静默替换。逐题ID、重复查询与观察hash统计见quality_metrics.json。

## 对后续工作的意义

动作格式、检索返回后接续回答、简短答案模仿有可用的冷启动效果。当前薄弱处集中在证据取舍、查询改写、停止策略和推理真实性。优先按既定计划验收RL初始化与训练接线，比继续扩大同一冷启动轨迹的epoch更合理；这是工程优先级判断，不是已经验证的RL收益。

现有终局答案reward对2090这种“答案对、解释错”的情况缺少反馈。转入RL也不会自动保证解释正确，应保留这个评价局限，当前不新增复杂的推理评分器。
