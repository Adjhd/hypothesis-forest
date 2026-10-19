# hypothesis-forest（中文说明）

假设森林的最小实现：智能体在分叉里提出假设，控制器把它们合并进主森林；证据、分裂、合并、迁移都是带触发条件的编辑，
写入一条只追加、带哈希链的日志。库本身约 800 行、无依赖，目标是能逐行读懂和调试。

* `hforest/scope.py`：作用域与两个作用域的关系（等价 / 覆盖 / 重叠 / 未知 / 不相交）。
* `hforest/forest.py`：森林本体、14 条生长规则、日志与重放。
* `hforest/merge.py`：分叉与合并；按内容键判定 new / refines / derived_from / alias / archive。
* `hforest/priors.py`：把森林转成带角色的先验卡片；只有授权过的假设才是 direct_prior。
* `hforest/promotion.py`：配对符号检验门控。

运行：`python examples/steel_demo.py`，测试：`python -m unittest discover tests`。

两条核心性质（测试中直接检查）：失败的实验不能否定假设；迁移不继承结论。规则表与设计约定见 `README.md`。
